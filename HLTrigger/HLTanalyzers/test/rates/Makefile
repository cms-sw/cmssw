ROOTLIBS   = $(shell root-config --libs)
ROOTCFLAGS = $(shell root-config --cflags)

CXX        = gcc
CXXFLAGS   = $(ROOTCFLAGS) -Wall -g
LD         = gcc
LDFLAGS    = -Wall -g
LIBS       = $(ROOTLIBS) -lEG

OHltRates: OHltRates.cpp OHltTree.h OHltTree.cpp OHltMenu.h OHltMenu.cpp
	$(CXX) $(CXXFLAGS) $(LIBS) OHltMenu.cpp OHltTree.cpp OHltRates.cpp -o OHltRates

all: OHltRates
