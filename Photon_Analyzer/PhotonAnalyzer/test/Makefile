objects = main.o
PhotonClass.exe : $(objects)
	g++ -o PhotonClass.exe $(objects) `root-config --libs`
main.o : main.C PhotonClass.h PhotonClass.C
	g++ -Wall -Wno-deprecated `root-config --cflags` main.C -c
clean :
	rm PhotonClass.exe $(objects)

