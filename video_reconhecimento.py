import numpy as np
import cv2
import face_recognition
import datetime

vc = cv2.VideoCapture('video/Jon Snow The Real North.mp4')
cv2.namedWindow('Jon Snow The Real North',cv2.WINDOW_AUTOSIZE)

largura_video = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
altura_video = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))

video_faces_detectadas = cv2.VideoWriter('saida/video_faces_detectadas.mp4', cv2.VideoWriter_fourcc('X','2','6','4'), 10, (largura_video,altura_video))

imagem = face_recognition.load_image_file('imagem/jon3.png')
face_encoding = face_recognition.face_encodings(imagem)[0]

faces_conhecidas = [
    face_encoding
]

nomes_faces_conhecidas = [
    'Jon Snow'
]

face_locations = []
face_encodings = []
face_names = []
processar_frame = True
intervalos_face_conhecida = []

segundos_anterior = 0

while vc.isOpened():

    ret, img = vc.read()
    segundos = int(vc.get(cv2.CAP_PROP_POS_MSEC)/1e3)
    tempo = str(datetime.timedelta(seconds=segundos))

    if ret == True:

        small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        if processar_frame:
            # Detecta todas as faces em uma imagem
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # Compara a imagem de busca com todos os rostos existentes na imagem atual.
                matches = face_recognition.compare_faces(faces_conhecidas, face_encoding, tolerance=0.50)
                name = None

                if True in matches:
                    first_match_index = matches.index(True)
                    name = nomes_faces_conhecidas[first_match_index]
                    
                    if tempo not in intervalos_face_conhecida and (segundos_anterior + 1) != segundos:
                        intervalos_face_conhecida.append(tempo)

                if name is not None:
                    face_names.append(name)

        processar_frame = not processar_frame

        if len(intervalos_face_conhecida) > 0 and len(face_names) == 0 and intervalos_face_conhecida.count(tempo) == 0 and (segundos_anterior + 1) != segundos:
            intervalos_face_conhecida.append(tempo)
            
            with open('resumo/relatorio_faces_presentes.txt', 'a+') as f:
                f.write(f'Jon Snow : tempo {intervalos_face_conhecida[0]} – {intervalos_face_conhecida[1]}\n')
                intervalos_face_conhecida.clear()

        if len(face_names) > 0:
            # Aplica a label nas faces encontradas
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                #Desenha um retangulo  em torno da face
                cv2.rectangle(img, (left, top), (right, bottom), (0, 127, 255), 2)

                #Inclui o nome da face identificada
                cv2.rectangle(img, (left, bottom - 25), (right, bottom), (0, 127, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(img, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        cv2.imshow('Jon Snow The Real North', img)
        video_faces_detectadas.write(img)

    else:
        break

    if cv2.waitKey(1) == 27:
            break  # esc para encerrar vídeo        

    segundos_anterior = segundos

vc.release()
video_faces_detectadas.release()
cv2.destroyAllWindows()