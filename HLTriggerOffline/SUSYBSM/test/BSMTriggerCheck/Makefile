CC=g++ -m32 

CFLAGS= -Wall -fPIC -I. -I$(ROOTSYS)/include -L.

.cpp.o:
	$(CC) -c $(CFLAGS) $< 
.cpp.a:
	$(CC) -c $(CFLAGS) $<
	ar r $@ $*.o; rm $*.o
.f.o:
	$(CC) -ff90 $(OPT) -c $< 
.f.a:
	$(CC) -ff90 $(OPT) -c $< 
	ar r $@ $*.o; rm $*.o

# Chooses whether to use ranlib - not if your system is a linux one
all: triggerComparison.x

# 
triggerComparison.x: Makefile triggerComparison.o 
	$(CC) -fsigned-char -Wstrict-prototypes -Wimplicit  -Wmissing-prototypes -Wunused \
	-o triggerComparison.x $(CFLAGS) triggerComparison.o \
	-L$(ROOTSYS)/lib $(shell root-config --glibs) \
	-rdynamic -lg2c -ldl -lm;
	touch triggerComparison.x

# clean
clean: 
	rm *.o *.x
