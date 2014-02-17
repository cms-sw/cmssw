#
f77=g77
cflags=-O -g -c
link=g77
libs=`cernlib -v pro geant pawlib graflib/X11 packlib mathlib kernlib`
linkopts=-v
INCS = /usr/include/

all: mixture

mixture: mixture.o
	$(link) mixture.o $(libs) $(linkopts) -o mixture

mixture.o: mixture.f
	$(f77) $(cflags) mixture.f 

clean:
	rm *.o
