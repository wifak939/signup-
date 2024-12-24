import cv2
import face_recognition
from face_rec import database_cr, face_detection, visulize_identity, draw_rec,save_faces_database

camera = cv2.VideoCapture(0)
faces_database,names = database_cr()
print(names)

while True :
    ret,image = camera.read()

    c = cv2.waitKey(33) % 256
    if c == ord(' '):
        print("Détection encours")
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_det = face_detection(image)
        print(face_det)

        if len(face_det) > 0:
            box, idx = face_det
            # print(box)
            image = draw_rec(image, box)
            image_face = image
            face_encodings = face_recognition.face_encodings(image_face, [box])

            if len(face_encodings) > 0:


                distance = face_recognition.face_distance(faces_database, face_encodings[0])
                print('distance:', distance)

                # verified = face_recognition.compare_faces(face_encodings,user_face,tolerance=0.55)
                verified = face_recognition.compare_faces(faces_database, face_encodings[0])
                print(verified)
                if True in verified:
                    name = names[verified.index(True)]

                    print('Bienvenu', name,'vous êtes déjà enregisté dans la base des données')
                    image = visulize_identity(image, name, box)
                    # open_door()
                else:
                    print('Vous n\'êtes pas dans la base des données.' )
                    # print("Vous n'êtes pas dans la base des données." )

                    faces_database.append(face_encodings[0])
                    user_name = input('Entrer votre nom: ')
                    image = visulize_identity(image, user_name, box)
                    names.append(user_name)

                    cv2.imwrite('image_signup.png',image)
                    save_faces_database(faces_database, names)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    print(len(faces_database))
                    print(names)

                    print('Visage enregisté avec succès.')
    cv2.imshow('image',image)
    cv2.waitKey(1)

