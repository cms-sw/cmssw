LIBS = $(shell root-config --libs)
INCS = -I$(shell root-config --incdir) -I.

WARN = -Wall -Wshadow

EXE = unbinnedScaling

all: $(EXE)

$(EXE) : % : %.o
	g++ -O2 -o $@ $+ $(LIBS) $(WARN)

%.o : %.cc
	g++ -o $@ $+ -c -O2 $(INCS) $(WARN) -std=c++11

clean:
	@rm -f *.o # objects
	@rm -f $(EXE)
	@rm -f *.cxx lib*h # remove rootcints generated source and header files
	@rm -f AutoDict_* *_h.d *_cc.d *_C.d # cint generated files

