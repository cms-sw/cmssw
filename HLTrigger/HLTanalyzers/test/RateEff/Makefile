ROOTLIBS   = $(shell root-config --libs) -lGenVector
ROOTCFLAGS = $(shell root-config --cflags)

CFGLIBS   = -L libconfig-1.3/.libs
CFGCFLAGS = -I libconfig-1.3

CXX        = g++
CXXFLAGS   = $(ROOTCFLAGS) -Wall -g -I$(shell scram tool info boost | grep '^INCLUDE' | cut -d= -f2)
LD         = gcc
LDFLAGS    = -Wall -g
LIBS       = -L$(CMSSW_RELEASE_BASE)/external/$(SCRAM_ARCH)/lib $(ROOTLIBS) -lEG $(CFGLIBS) -lconfig++


OHltRateEff: OHltRateEff.cpp OHltConfig.cpp OHltConfig.h OHltMenu.cpp OHltMenu.h \
	 OHltRateCounter.cpp OHltRateCounter.h OHltRatePrinter.cpp OHltRatePrinter.h \
	 OHltEffPrinter.cpp OHltEffPrinter.h  PlotOHltEffCurves.cpp\
	 OHltTree.cpp OHltTreeOpen.cpp OHltTree.h OHltPileupRateFitter.cpp OHltPileupRateFitter.h L1GtLogicParser.cpp L1GtLogicParser.h HLTDatasets.h
	$(CXX) $(CXXFLAGS) $(CFGCFLAGS) $(LIBS) OHltConfig.cpp  OHltMenu.cpp OHltRateEff.cpp \
	OHltEffPrinter.cpp OHltEffPrinter.h PlotOHltEffCurves.cpp \
	OHltRateCounter.cpp OHltRatePrinter.cpp OHltTree.cpp OHltTreeOpen.cpp L1GtLogicParser.cpp HLTDatasets.cpp OHltPileupRateFitter.cpp \
	-o OHltRateEff

all: OHltRateEff
