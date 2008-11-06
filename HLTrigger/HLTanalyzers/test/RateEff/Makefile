ROOTLIBS   = $(shell root-config --libs)
ROOTCFLAGS = $(shell root-config --cflags)

CXX        = gcc
CXXFLAGS   = $(ROOTCFLAGS) -Wall -g
LD         = gcc
LDFLAGS    = -Wall -g
LIBS       = $(ROOTLIBS) -lEG

OHltRateEff: OHltRateEff.cpp OHltTree.h OHltTree.cpp OHltMenu.h OHltMenu.cpp 
	$(CXX) $(CXXFLAGS) $(LIBS) OHltMenu.cpp OHltTree.cpp OHltRateEff.cpp -o OHltRateEff

all: OHltRates
