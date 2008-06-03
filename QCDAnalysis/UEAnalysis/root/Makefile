objects = UEAnalysis.o UEAnalysisOnRootple.o \
          UEAnalysisWeight.o  UEAnalysisCorrCali.o \
          UEAnalysisJets.o  UEAnalysisAntiKtJets.o \
          UEAnalysisMPI.o  UEAnalysisUE.o  
source1 = UEAnalysis.cpp
source2 = UEAnalysisOnRootple.C
source3 = UEAnalysisWeight.C
source4 = UEAnalysisCorrCali.C
source5 = UEAnalysisJets.C
source6 = UEAnalysisAntiKtJets.C
source7 = UEAnalysisMPI.C
source8 = UEAnalysisUE.C

PREFIX        = /afs/cern.ch/sw/lcg/external
SYSNAME       = slc4_ia32_gcc34
FastJetdir    = $(PREFIX)/fastjet/2.3.3/$(SYSNAME)
FastJetlib    = -L$(FastJetdir)/lib -lfastjet -lsiscone -l SISConePlugin

CXX           = g++
CXXFLAGS      = -O -Wall -fPIC
LD            = g++
ROOTCFLAGS   := $(shell root-config --cflags)
ROOTLIBS     := $(shell root-config --libs)
ROOTGLIBS    := $(shell $(ROOTSYS)/bin/root-config --glibs)
CXXFLAGS     += $(ROOTCFLAGS)
CXXFLAGS     += -I$(FastJetdir)/include
LIBS          = $(ROOTLIBS) -lEG $(FastJetlib)
SHARE         = -shared
NAMELIB       = $(CMSSW_BASE)/src/QCDAnalysis/UEAnalysis/root/UE.o
BINFILE       = UEAnalysis
lib :
	$(LD) -g $(ROOTCFLAGS) -I$(FastJetdir)/include -c $(source1)
	$(LD) -g $(ROOTCFLAGS) -I$(FastJetdir)/include -c $(source2)
	$(LD) -g $(ROOTCFLAGS) -I$(FastJetdir)/include -c $(source3)
	$(LD) -g $(ROOTCFLAGS) -I$(FastJetdir)/include -c $(source4)
	$(LD) -g $(ROOTCFLAGS) -I$(FastJetdir)/include -c $(source5)
	$(LD) -g $(ROOTCFLAGS) -I$(FastJetdir)/include -c $(source6)
	$(LD) -g $(ROOTCFLAGS) -I$(FastJetdir)/include -c $(source7)
	$(LD) -g $(ROOTCFLAGS) -I$(FastJetdir)/include -c $(source8)
	$(LD) -g $(objects) $(LIBS) $(SHARE) -o $(NAMELIB) $(FastJetlib)
 
bin:
	$(LD) -g $(LIBS) $(NAMELIB) -o $(BINFILE)
 
 
clean:
	rm $(objects) $(BINFILE) $(NAMELIB) G__a*
