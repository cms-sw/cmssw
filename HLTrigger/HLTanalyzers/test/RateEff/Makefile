ROOTLIBS   = $(shell root-config --libs)
ROOTCFLAGS = $(shell root-config --cflags)

CFGLIBS   = -L$(shell pwd)/libconfig-1.3/.libs
CFGCFLAGS = -I$(shell pwd)/libconfig-1.3

CXX        = g++
CXXFLAGS   = $(ROOTCFLAGS) -Wall -g
LD         = gcc
LDFLAGS    = -Wall -g
LIBS       = $(ROOTLIBS) -lEG $(CFGLIBS) -lconfig++

OHltRateEff: OHltRateEff.cpp OHltConfig.cpp OHltConfig.h OHltMenu.cpp OHltMenu.h \
	 OHltRateCounter.cpp OHltRateCounter.h OHltRatePrinter.cpp OHltRatePrinter.h \
	 OHltEffPrinter.cpp OHltEffPrinter.h  PlotOHltEffCurves.cpp\
	 OHltTree.cpp OHltTreeOpen.cpp OHltTree.h L1GtLogicParser.cpp L1GtLogicParser.h HLTDatasets.h
	$(CXX) $(CXXFLAGS) $(CFGCFLAGS) $(LIBS) OHltConfig.cpp  OHltMenu.cpp OHltRateEff.cpp \
	OHltEffPrinter.cpp OHltEffPrinter.h PlotOHltEffCurves.cpp \
	OHltRateCounter.cpp OHltRatePrinter.cpp OHltTree.cpp OHltTreeOpen.cpp L1GtLogicParser.cpp HLTDatasets.cpp \
	-o OHltRateEff

all: OHltRateEff
