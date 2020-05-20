#common definitions go in here
CXX           = g++
CXXFLAGS      = -O -Wall  -fPIC -g -ansi -Wextra -Wno-deprecated -std=c++11 -I. -I../../..
LD            = g++
LDFLAGS       = -O
LIBDIR       = $(WORKDIR)
AR            = ar
ARFLAGS       = cr

OS := $(shell uname -s)
ifeq ($(OS),Darwin)
	SOFLAGS       = -dynamiclib
else
	SOFLAGS       = -shared
endif	

ifndef ROOTSYS
$(error *** Please set up Root)
endif

ifeq ("$(wildcard lib)","")
$(shell mkdir lib)
endif

#
# Add system headers here (hack around problem with makedepend
# if you added file remove the dummyIncl directory to rewrite files
#
ifeq ("$(wildcard dummyIncl)","")
$(shell mkdir dummyIncl)
$(shell mkdir dummyIncl/sys)
$(shell mkdir dummyIncl/ext)
$(shell touch dummyIncl/TBranch.h)
$(shell touch dummyIncl/TCanvas.h)
$(shell touch dummyIncl/TChain.h)
$(shell touch dummyIncl/TColor.h)
$(shell touch dummyIncl/TEfficiency.h)
$(shell touch dummyIncl/TFile.h)
$(shell touch dummyIncl/TH1D.h)
$(shell touch dummyIncl/TH1F.h)
$(shell touch dummyIncl/TH2F.h)
$(shell touch dummyIncl/TLatex.h)
$(shell touch dummyIncl/TLegend.h)
$(shell touch dummyIncl/TMath.h)
$(shell touch dummyIncl/TROOT.h)
$(shell touch dummyIncl/TStyle.h)
$(shell touch dummyIncl/TSystem.h)
$(shell touch dummyIncl/TTree.h)
$(shell touch dummyIncl/algorithm)
$(shell touch dummyIncl/array)
$(shell touch dummyIncl/bitset)
$(shell touch dummyIncl/cassert)
$(shell touch dummyIncl/chrono)
$(shell touch dummyIncl/cctype)
$(shell touch dummyIncl/cmath)
$(shell touch dummyIncl/ctime)
$(shell touch dummyIncl/cstdlib)
$(shell touch dummyIncl/cmath)
$(shell touch dummyIncl/exception)
$(shell touch dummyIncl/fstream)
$(shell touch dummyIncl/iomanip)
$(shell touch dummyIncl/iostream)
$(shell touch dummyIncl/limits)
$(shell touch dummyIncl/map)
$(shell touch dummyIncl/memory)
$(shell touch dummyIncl/set)
$(shell touch dummyIncl/sstream)
$(shell touch dummyIncl/string)
$(shell touch dummyIncl/vector)
$(shell touch dummyIncl/unordered_map)
$(shell touch dummyIncl/unordered_set)
$(shell touch dummyIncl/utility)
$(shell touch dummyIncl/algorithm)
$(shell touch dummyIncl/ext/hash_set)
$(shell touch dummyIncl/sys/time.h)
$(shell touch dummyIncl/ctime)
$(shell touch dummyIncl/chrono)
endif



ifeq ("$(wildcard FWCore)","")
$(shell mkdir FWCore)
$(shell mkdir FWCore/MessageLogger)
$(shell mkdir FWCore/MessageLogger/interface)
$(shell mkdir FWCore/Utilities)
$(shell mkdir FWCore/Utilities/interface)
endif

ifeq ("$(wildcard FWCore/Utilities/interface/Exception.h)","")
$(shell echo "#ifndef L1Trigger_TrackFindingTracklet_interface_Exception_h" > FWCore/Utilities/interface/Exception.h)
$(shell echo "#define L1Trigger_TrackFindingTracklet_interface_Exception_h" >> FWCore/Utilities/interface/Exception.h)
$(shell echo "#include<exception>" >> FWCore/Utilities/interface/Exception.h)
$(shell echo "namespace cms {" >> FWCore/Utilities/interface/Exception.h)
$(shell echo "  class Exception{" >> FWCore/Utilities/interface/Exception.h)
$(shell echo "  public:" >> FWCore/Utilities/interface/Exception.h)
$(shell echo "    Exception(std::string type) { if(0) {std::cout<<type<<std::endl;} }" >> FWCore/Utilities/interface/Exception.h)
$(shell echo "    ~Exception() { std::cout << std::endl;}" >> FWCore/Utilities/interface/Exception.h)
$(shell echo "    template <class T> Exception& operator<<(T const& t) {std::cout << t;return *this;}" >> FWCore/Utilities/interface/Exception.h)
$(shell echo "    Exception& operator<<(std::ostream& (*f)(std::ostream&)) {std::cout << f;return *this;}" >> FWCore/Utilities/interface/Exception.h)
$(shell echo "    Exception& operator<<(std::ios_base& (*f)(std::ios_base&)) {std::cout << f;return *this;}" >> FWCore/Utilities/interface/Exception.h)
$(shell echo "  };" >> FWCore/Utilities/interface/Exception.h)
$(shell echo "};" >> FWCore/Utilities/interface/Exception.h)
$(shell echo "#endif" >> FWCore/Utilities/interface/Exception.h)
endif

ifeq ("$(wildcard FWCore/MessageLogger/interface/MessageLogger.h)","")
$(shell echo "#ifndef L1Trigger_TrackFindingTracklet_interface_Logger_h" > FWCore/MessageLogger/interface/MessageLogger.h)
$(shell echo "#define L1Trigger_TrackFindingTracklet_interface_Logger_h" >> FWCore/MessageLogger/interface/MessageLogger.h)
$(shell echo "#include<iostream>" >> FWCore/MessageLogger/interface/MessageLogger.h)
$(shell echo "namespace edm {" >> FWCore/MessageLogger/interface/MessageLogger.h)
# INFO statement
$(shell echo "  class LogVerbatim{" >> FWCore/MessageLogger/interface/MessageLogger.h)
$(shell echo "  public:" >> FWCore/MessageLogger/interface/MessageLogger.h)
$(shell echo "    LogVerbatim(std::string type) { if(0) {std::cout<<type<<std::endl;} }" >> FWCore/MessageLogger/interface/MessageLogger.h)
$(shell echo "    ~LogVerbatim() { std::cout << std::endl;}" >> FWCore/MessageLogger/interface/MessageLogger.h)
$(shell echo "    template <class T> LogVerbatim& operator<<(T const& t) {std::cout << t;return *this;}" >> FWCore/MessageLogger/interface/MessageLogger.h)
$(shell echo "    LogVerbatim& operator<<(std::ostream& (*f)(std::ostream&)) {std::cout << f;return *this;}" >> FWCore/MessageLogger/interface/MessageLogger.h)
$(shell echo "    LogVerbatim& operator<<(std::ios_base& (*f)(std::ios_base&)) {std::cout << f;return *this;}" >> FWCore/MessageLogger/interface/MessageLogger.h)
$(shell echo "  };" >> FWCore/MessageLogger/interface/MessageLogger.h)
# WARNING statement
$(shell echo "  class LogPrint{" >> FWCore/MessageLogger/interface/MessageLogger.h)
$(shell echo "  public:" >> FWCore/MessageLogger/interface/MessageLogger.h)
$(shell echo "    LogPrint(std::string type) { if(0) {std::cout<<type<<std::endl;} }" >> FWCore/MessageLogger/interface/MessageLogger.h)
$(shell echo "    ~LogPrint() { std::cout << std::endl;}" >> FWCore/MessageLogger/interface/MessageLogger.h)
$(shell echo "    template <class T> LogPrint& operator<<(T const& t) {std::cout << t;return *this;}" >> FWCore/MessageLogger/interface/MessageLogger.h)
$(shell echo "    LogPrint& operator<<(std::ostream& (*f)(std::ostream&)) {std::cout << f;return *this;}" >> FWCore/MessageLogger/interface/MessageLogger.h)
$(shell echo "    LogPrint& operator<<(std::ios_base& (*f)(std::ios_base&)) {std::cout << f;return *this;}" >> FWCore/MessageLogger/interface/MessageLogger.h)
$(shell echo "  };" >> FWCore/MessageLogger/interface/MessageLogger.h)
# another WARNING statement
$(shell echo "  class LogWarning{" >> FWCore/MessageLogger/interface/MessageLogger.h)
$(shell echo "  public:" >> FWCore/MessageLogger/interface/MessageLogger.h)
$(shell echo "    LogWarning(std::string type) { if(0) {std::cout<<type<<std::endl;} }" >> FWCore/MessageLogger/interface/MessageLogger.h)
$(shell echo "    ~LogWarning() { std::cout << std::endl;}" >> FWCore/MessageLogger/interface/MessageLogger.h)
$(shell echo "    template <class T> LogWarning& operator<<(T const& t) {std::cout << t;return *this;}" >> FWCore/MessageLogger/interface/MessageLogger.h)
$(shell echo "    LogWarning& operator<<(std::ostream& (*f)(std::ostream&)) {std::cout << f;return *this;}" >> FWCore/MessageLogger/interface/MessageLogger.h)
$(shell echo "    LogWarning& operator<<(std::ios_base& (*f)(std::ios_base&)) {std::cout << f;return *this;}" >> FWCore/MessageLogger/interface/MessageLogger.h)
$(shell echo "  };" >> FWCore/MessageLogger/interface/MessageLogger.h)
# ERROR statement
$(shell echo "  class LogProblem{" >> FWCore/MessageLogger/interface/MessageLogger.h)
$(shell echo "  public:" >> FWCore/MessageLogger/interface/MessageLogger.h)
$(shell echo "    LogProblem(std::string type) { if(0) {std::cout<<type<<std::endl;} }" >> FWCore/MessageLogger/interface/MessageLogger.h)
$(shell echo "    ~LogProblem() { std::cout << std::endl;}" >> FWCore/MessageLogger/interface/MessageLogger.h)
$(shell echo "    template <class T> LogProblem& operator<<(T const& t) {std::cout << t;return *this;}" >> FWCore/MessageLogger/interface/MessageLogger.h)
$(shell echo "    LogProblem& operator<<(std::ostream& (*f)(std::ostream&)) {std::cout << f;return *this;}" >> FWCore/MessageLogger/interface/MessageLogger.h)
$(shell echo "    LogProblem& operator<<(std::ios_base& (*f)(std::ios_base&)) {std::cout << f;return *this;}" >> FWCore/MessageLogger/interface/MessageLogger.h)
$(shell echo "  };" >> FWCore/MessageLogger/interface/MessageLogger.h)
#
$(shell echo "};" >> FWCore/MessageLogger/interface/MessageLogger.h)
$(shell echo "#endif" >> FWCore/MessageLogger/interface/MessageLogger.h)
endif



#find out about the installed ROOT
ROOTCONFIG   := $(ROOTSYS)/bin/root-config
ROOTCFLAGS   := $(shell $(ROOTCONFIG) --cflags)
ROOTLDFLAGS  := $(shell $(ROOTCONFIG) --ldflags) $(shell $(ROOTCONFIG) --libs --nonew --glibs)

#append to options
CXXFLAGS     += $(ROOTCFLAGS)
LDFLAGS      += $(ROOTLDFLAGS)
# Needed for ROOT-Tree
#LDFLAGS      += -L./ FPGAEvent_cxx.so


LIBS = -lHtml


ifneq ($(CMSSW_BASE),)
	ROOTFIT_INCLUDE := $(shell cd $(CMSSW_BASE); scram tool info roofitcore | grep INCLUDE= | sed 's|INCLUDE=||')
	ROOTFIT_LIBDIR := $(shell cd $(CMSSW_BASE); scram tool info roofitcore | grep LIBDIR= | sed 's|LIBDIR=||')

	CINTINCLUDES := -I$(ROOTFIT_INCLUDE)
	CXXFLAGS     += -I$(ROOTFIT_INCLUDE)
	LDFLAGS      += -L$(ROOTFIT_LIBDIR)
endif


DEPDIR = .deps

MAKEDEPEND = makedepend -I../../.. -IdummyIncl -f- $< | sed 's/..\/src/lib/' > .deps/$*.d

SRCS = $(wildcard ../src/*.cc)

OBJS = $(SRCS:../src/%.cc=lib/%.o)

default: fpga

fpga: fpga.a fpga.o
	$(LD) -o $@ fpga.o fpga.a $(LDFLAGS) $(LIBS)

fpga.o: fpga.cc $(DEPDIR)/fpga.d | $(DEPDIR)
	@$(MAKEDEPEND)
	$(CXX) $(CXXFLAGS) -c fpga.cc -o $@

fpga.a:$(OBJS) 
	@$(AR) $(ARFLAGS) $@ $?

lib/%.o:../src/%.cc $(DEPDIR)/%.d | $(DEPDIR)
	@$(MAKEDEPEND)
	time $(CXX) $(CXXFLAGS) -o $@ -c $<

$(DEPDIR): ; @mkdir -p $@

%.d: ;

DEPFILES := $(SRCS:../src/%.cc=$(DEPDIR)/%.d)
$(DEPFILES):
include $(wildcard $(DEPFILES))
include $(wildcard $(DEPDIR)/fpga.d)


clean:
	@rm -f *.o
	@rm -f matching
	@rm -f main
	@rm -f fpga
	@rm -f fpga.a
	@rm -f *~
	@rm -f ../data/MemPrints/InputStubs/InputStubs*.dat
	@rm -f ../data/MemPrints/InputStubs/Link_*.dat
	@rm -f ../data/MemPrints/Stubs/AllStubs_AS_*.dat
	@rm -f ../data/MemPrints/Stubs/LayerStubs_SL*.dat
	@rm -f ../data/MemPrints/VMStubsTE/VMStubs_VMS*.dat
	@rm -f ../data/MemPrints/VMStubsME/VMStubs_VMS*.dat
	@rm -f ../data/MemPrints/FitTrack/TrackFit_TF_*.dat
	@rm -f ../data/MemPrints/CleanTrack/CleanTrack_CT_*.dat
	@rm -f ../data/MemPrints/Stubs/DiskStubs_SD*.dat
	@rm -f ../data/MemPrints/StubPairs/StubPairs_SP_*.dat
	@rm -f ../data/MemPrints/StubPairs/StubPairs_SPD_*.dat
	@rm -f ../data/MemPrints/StubPairs/StubTriplets_ST_*.dat
	@rm -f InvTable_TC_*.dat
	@rm -f FitDerTable_*.txt
	@rm -f TEBinTable*.txt
	@rm -f ../data/MemPrints/TrackletProjections/TrackletProjections_TPROJ_*.dat
	@rm -f ../data/MemPrints/TrackletProjections/AllProj_AP_*.dat
	@rm -f ../data/MemPrints/VMProjections/VMProjections_VMPROJ_*.dat
	@rm -f ../data/MemPrints/Matches/FullMatches_FM_*.dat
	@rm -f ../data/MemPrints/Matches/CandidateMatches_CM_*.dat
	@rm -f ../data/MemPrints/TrackletParameters/TrackletParameters_TPAR_*.dat
	@rm -f layeroccupancy.txt
	@rm -f VMTable*.tab
	@rm -f METable_*.tab
	@rm -f *_vmbendcut.tab
	@rm -f *_finebin.tab
	@rm -f *_phicut.tab
	@rm -f *_zcut.tab
	@rm -f *_PSphicut.tab
	@rm -f *_2Sphicut.tab
	@rm -f *_PSrcut.tab
	@rm -f *_2Srcut.tab
	@rm -f *_PSzcut.tab
	@rm -f *_2Szcut.tab
	@rm -f *_ptcut.tab
	@rm -f *_stubptinnercut.tab
	@rm -f *_stubptoutercut.tab
	@rm -f TE_*_*_*.txt
	@rm -f moduleoccupancy.txt
	@rm -f trackletcombinations.txt
	@rm -f trackletlayercombinations.txt
	@rm -f trackletlayers.txt
	@rm -f trackletparameters.txt
	@rm -f tracklets.txt
	@rm -f trackletpars.txt
	@rm -f trackletparsdisk.txt
	@rm -f trackproj.txt
	@rm -f diskmatch.txt
	@rm -f neighborproj.txt
	@rm -f vmfullmatch.txt
	@rm -f vmmatches.txt
	@rm -f vmstubs.txt
	@rm -f allprojections.txt
	@rm -f stubs.txt
	@rm -f stubsdisk.txt
	@rm -f stubslayer.txt
	@rm -f stubpairs.txt
	@rm -f stubpairsdisk.txt
	@rm -f trackletcands.txt
	@rm -f trackletslayer.txt
	@rm -f trackres.txt
	@rm -f vmprojections.txt
	@rm -f trackeff.txt
	@rm -f allstubs.txt
	@rm -f diskrouter.txt
	@rm -f fittrack.txt
	@rm -f fittrack1.txt
	@rm -f layerrouter.txt
	@rm -f matchcalculator.txt
	@rm -f projectioncounts.txt
	@rm -f alphadep.txt
	@rm -f diskmatch*.txt
	@rm -f dtclinks.txt
	@rm -f vmoccupancy.txt
	@rm -f vmoccupancyme.txt
	@rm -f vmoccupancyte.txt
	@rm -f layerresiduals.txt
	@rm -f diskresiduals.txt
	@rm -f chisq.txt
	@rm -f VMPhiCorrL*.txt
	@rm -f newvmoccupancy.txt
	@rm -f nmatches*.txt
	@rm -f hitpattern.txt
	@rm -f matchengine.txt
	@rm -f stubslayerpersector.txt
	@rm -f stubsdiskpersector.txt
	@rm -f z0_and_rinv.txt
	@rm -f trackletengine.txt
	@rm -f trackletcalculator.txt
	@rm -f matchtransceiver.txt
	@rm -f matchtransceiver_cc*
	@rm -f projectiontransceiver.txt
	@rm -f projectiontransceiver_cc*
	@rm -f trackletcalculator.txt
	@rm -f allstubste.txt
	@rm -f allstubsme.txt
	@rm -f variance.txt
	@rm -f hiteff.txt
	@rm -f dtcphirange.txt
	@rm -f inputlink.txt
	@rm -f unusedproj.txt
	@rm -f trackprojocc.txt
	@rm -f trackprojocc.dat
	@rm -f trackpars.txt
	@rm -f AutoDict_*.cxx
	@rm -f AutoDict_*cxx.d
	@rm -f AutoDict_*cxx.so
	@rm -f *_drinv.tab
	@rm -f *_drinv.h
	@rm -f *_tinv.tab
	@rm -f *_tinv.h
	@rm -f *_invt.tab
	@rm -f *_invt.h
	@rm -f matcheff.txt
	@rm -f TC_*.v
	@rm -f TC_*.cpp
	@rm -f imath.root
	@rm -f *_cc.d
	@rm -f *_cc.so
	@rm -f *_dict_rdict.pcm
	@rm -f imath.a
	@rm -rf FWCore/
	@rm -rf lib/
	@rm -rf dummyIncl/
	@rm -rf .deps/



# DO NOT DELETE THIS LINE - make depend depends on it.


