CXXFLAGS:=$(shell root-config --cflags) -ggdb -O2 -I$(BASEDIR) -fPIC
LDFLAGS:=$(shell root-config --ldflags --libs)  -lRooFitCore -lRooStats -lRooFit -lFoam -lMinuit -lstdc++  -fPIC

SOURCES:=$(wildcard *.cxx)
EXES=$(SOURCES:%.cxx=%.exe)

.PHONY: all
all: $(EXES)

$(EXES): %.exe: %.cxx
	gcc $(CXXFLAGS) $(LDFLAGS) $< -o $@

.PHONY: clean
clean:
	rm *.exe

