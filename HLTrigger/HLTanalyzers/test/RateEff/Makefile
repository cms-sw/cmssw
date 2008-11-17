ROOTLIBS   = $(shell root-config --libs)
ROOTCFLAGS = $(shell root-config --cflags)

CXX        = gcc
CXXFLAGS   = $(ROOTCFLAGS) -Wall -g
#CXXFLAGS  += -DUSE_SAMPLES_AT_FNAL
LD         = gcc
LDFLAGS    = -Wall -g
LIBS       = $(ROOTLIBS) -lEG

OHltRateEff: OHltRateEff.cpp OHltTree.h OHltTree.cpp OHltMenu.h OHltMenu.cpp HLTDatasets.cpp HLTDatasets.h
	$(CXX) $(CXXFLAGS) $(LIBS) OHltMenu.cpp OHltTree.cpp OHltRateEff.cpp HLTDatasets.cpp -o OHltRateEff

all: OHltRates
