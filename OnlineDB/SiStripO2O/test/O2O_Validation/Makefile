CC                              =g++
CFLAGS                          = -Wall -O -Wno-deprecated
#INCLUDE                         =-I$(ROOTSYS)/include

#ROOT-------------------------------------
ROOTCFLAGS			=$(shell root-config --cflags)
ROOTLIBS			=$(shell root-config --libs)
ROOTGLIBS			=$(shell root-config --glibs)
#-----------------------------------------

LIBS				=$(ROOTLIBS)

FILES				= pednoise_validation.cc

#HEADERS                         =$(addsuffix .h, $(FILES))
#OBJECTS                         =$(addsuffix .o, $(FILES))

all:				pednoise_validation

pednoise_validation:		$(OBJECTS) pednoise_validation.cc
				$(CC) $(CFLAGS) $(ROOTCFLAGS) $(LIBS) -o pednoise_validation pednoise_validation.cc $(OBJECTS)

clean:
				@rm -rf *~ *.o *Dict*

#distclean:			clean
#				@rm -rf mckammer

