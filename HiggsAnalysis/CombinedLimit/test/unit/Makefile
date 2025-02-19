CXXFLAGS:=$(shell root-config --cflags) -ggdb -O2 -I$(ROOFITSYS)/include -I $(CMSSW_BASE)/src -I $(CMSSW_BASE)/src/HiggsAnalysis/CombinedLimit/src -fPIC
LDFLAGS:=$(shell root-config --ldflags --libs) -L$(ROOFITSYS)/lib -L $(CMSSW_BASE)/lib/$(SCRAM_ARCH) -lRooFitCore -lRooStats -lRooFit -lFoam -lMinuit -lstdc++ -lHiggsAnalysisCombinedLimit -fPIC

SOURCES:=$(wildcard *.cxx)
EXES=$(SOURCES:%.cxx=%.exe)

.PHONY: all
all: $(EXES)

$(EXES): %.exe: %.cxx
	gcc $(CXXFLAGS) $(LDFLAGS) $< -o $@

.PHONY: clean
clean:
	rm *.exe

