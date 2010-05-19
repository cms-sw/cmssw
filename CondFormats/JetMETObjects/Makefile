################################################################################
#
# CondFormats/JetMETObjects Makefile (for standalone use outside CMSSW/SCRAM)
# ---------------------------------------------------------------------------
#
# INSTRUCTIONS:
# =============
# setenv ROOTSYS /path/to/root
# setenv PATH $ROOTSYS/bin:${PATH}
# setenv LD_LIBRARY_PATH $ROOTSYS/lib
#
# mkdir standalone; cd standalone
# setenv STANDALONE_DIR $PWD
# setenv PATH $STANDALONE_DIR/bin:${PATH}
# setenv LD_LIBRARY_PATH $STANDALONE_DIR/lib:${LD_LIBRARY_PATH}
# cvs co -d JetMETObjects CMSSW/CondFormats/JetMETObjects
# cd JetMETObjects
# make
#
# [you might want to stick these into e.g. $STANDALONE_DIR/setup.[c]sh]
#
#             07/11/2009 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
################################################################################

ifeq ($(STANDALONE_DIR),)
	standalone_dir:=../
	export STANDALONE_DIR:=$(standalone_dir)
endif


TMPDIR       = $(STANDALONE_DIR)/tmp
LIBDIR       = $(STANDALONE_DIR)/lib
BINDIR       = $(STANDALONE_DIR)/bin


CXX          = g++


ROOTCXXFLAGS = $(shell $(ROOTSYS)/bin/root-config --cflags)
CXXFLAGS     = -O3 -Wall -fPIC -DSTANDALONE -I. $(ROOTCXXFLAGS)

ROOTLIBS     = $(shell $(ROOTSYS)/bin/root-config --libs)

OBJS         = $(TMPDIR)/JetCorrectorParameters.o \
	       $(TMPDIR)/SimpleJetCorrector.o \
               $(TMPDIR)/FactorizedJetCorrector.o \
               $(TMPDIR)/JetResolution.o \
               $(TMPDIR)/JetMETObjects_dict.o

LIB          = libJetMETObjects.so


all: setup lib

test: test_JetCorrectorParameters test_JetResolution

setup:
	rm -f CondFormats; ln -sf ../ CondFormats
	mkdir -p $(TMPDIR)
	mkdir -p $(LIBDIR)
	mkdir -p $(BINDIR)

lib: $(OBJS)
	$(CXX) $(CXXFLAGS) -shared $(OBJS) $(ROOTLIBS) -o $(LIBDIR)/$(LIB)

test_JetCorrectorParameters: lib
	$(CXX) $(CXXFLAGS) -L$(LIBDIR) -lJetMETObjects $(ROOTLIBS) \
	bin/JetCorrectorParameters_t.cc -o $(BINDIR)/JetCorrectorParameters_t

test_JetResolution: lib
	$(CXX) $(CXXFLAGS) -L$(LIBDIR) -lJetMETObjects $(ROOTLIBS) \
	bin/JetResolution_t.cc -o $(BINDIR)/JetResolution_t


clean:
	rm -rf $(OBJS) $(LIBDIR)/$(LIB) CondFormats \
	       $(TMPDIR)/JetMETObjects_dict.h $(TMPDIR)/JetMETObjects_dict.cc \
               $(BINDIR)/JetResolution_t $(BINDIR)/JetCorrectorParameters_t


################################################################################
# $(OBJS)
################################################################################

$(TMPDIR)/JetCorrectorParameters.o: interface/JetCorrectorParameters.h \
				    src/JetCorrectorParameters.cc
	$(CXX) $(CXXFLAGS) -c src/JetCorrectorParameters.cc \
	-o $(TMPDIR)/JetCorrectorParameters.o 

$(TMPDIR)/SimpleJetCorrector.o: interface/SimpleJetCorrector.h \
				    src/SimpleJetCorrector.cc
	$(CXX) $(CXXFLAGS) -c src/SimpleJetCorrector.cc \
	-o $(TMPDIR)/SimpleJetCorrector.o 

$(TMPDIR)/FactorizedJetCorrector.o: interface/FactorizedJetCorrector.h \
				    src/FactorizedJetCorrector.cc
	$(CXX) $(CXXFLAGS) -c src/FactorizedJetCorrector.cc \
	-o $(TMPDIR)/FactorizedJetCorrector.o 

$(TMPDIR)/JetResolution.o: interface/JetResolution.h \
		           src/JetResolution.cc
	$(CXX) $(CXXFLAGS) -c src/JetResolution.cc \
	-o $(TMPDIR)/JetResolution.o 




$(TMPDIR)/JetMETObjects_dict.o: $(TMPDIR)/JetMETObjects_dict.cc
	$(CXX) $(CXXFLAGS) -I$(TMPDIR) -c $(TMPDIR)/JetMETObjects_dict.cc \
	-o $(TMPDIR)/JetMETObjects_dict.o

$(TMPDIR)/JetMETObjects_dict.cc: interface/JetCorrectorParameters.h \
				 interface/SimpleJetCorrector.h \
				 interface/JetResolution.h \
				 interface/FactorizedJetCorrector.h \
				 interface/Linkdef.h
	rm -rf $(TMPDIR)/JetMETObjects_dict.h
	rm -rf $(TMPDIR)/JetMETObjects_dict.cc
	$(ROOTSYS)/bin/rootcint -f $(TMPDIR)/JetMETObjects_dict.cc \
	-c -I$(TMPDIR) \
	interface/JetCorrectorParameters.h \
	interface/JetResolution.h \
	interface/SimpleJetCorrector.h \
	interface/FactorizedJetCorrector.h \
	interface/Linkdef.h
