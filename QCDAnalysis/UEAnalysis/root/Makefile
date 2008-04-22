objects = UEAnalysis.o UEAnalysisOnRootple.o UEAnalysisWeight.o  UEAnalysisCorrCali.o  UEAnalysisJets.o  UEAnalysisMPI.o  UEAnalysisUE.o  
source1 = UEAnalysis.cpp
source2 = UEAnalysisOnRootple.C
source3 = UEAnalysisWeight.C
source4 = UEAnalysisCorrCali.C
source5 = UEAnalysisJets.C
source6 = UEAnalysisMPI.C
source7 = UEAnalysisUE.C

CXX           = g++
CXXFLAGS      = -O -Wall -fPIC
LD            = g++
ROOTCFLAGS   := $(shell root-config --cflags)
ROOTLIBS     := $(shell root-config --libs)
ROOTGLIBS    := $(shell $(ROOTSYS)/bin/root-config --glibs)
CXXFLAGS     += $(ROOTCFLAGS)
LIBS          = $(ROOTLIBS) -lEG 
SHARE         = -shared
NAMELIB       = $(CMSSW_BASE)/src/QCDAnalysis/UEAnalysis/root/UE.o
BINFILE       = UEAnalysis
lib :
	$(LD) -g $(ROOTCFLAGS) -c $(source1)
	$(LD) -g $(ROOTCFLAGS) -c $(source2)
	$(LD) -g $(ROOTCFLAGS) -c $(source3)
	$(LD) -g $(ROOTCFLAGS) -c $(source4)
	$(LD) -g $(ROOTCFLAGS) -c $(source5)
	$(LD) -g $(ROOTCFLAGS) -c $(source6)
	$(LD) -g $(ROOTCFLAGS) -c $(source7)
	$(LD) -g $(objects) $(LIBS) $(SHARE) -o $(NAMELIB)
 
bin:
	$(LD) -g $(LIBS) $(NAMELIB) -o $(BINFILE)
 
 
clean:
	rm $(objects) $(BINFILE) $(NAMELIB) G__a*
