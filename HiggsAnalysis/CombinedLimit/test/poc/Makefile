ROOFITDIR=$(shell scramv1 tool tag roofitcore ROOFIT_BASE)
CXXFLAGS:=$(shell root-config --cflags) -ggdb -O2 -I$(ROOFITDIR)/include -fPIC
LDFLAGS:=$(shell root-config --ldflags --libs) -L$(ROOFITDIR)/lib  -lRooFitCore -lRooStats -lRooFit -lFoam -lMinuit -lstdc++  -fPIC

SOURCES:=$(wildcard *.cxx)
EXES=$(SOURCES:%.cxx=%.exe)

.PHONY: all
all: $(EXES)

$(EXES): %.exe: %.cxx
	gcc $(CXXFLAGS) $(LDFLAGS) $< -o $@

.PHONY: clean
clean:
	rm *.exe

