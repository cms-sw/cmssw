#common definitions go in here
CXX           = g++
CXXFLAGS      = -O -Wall  -fPIC -g -ansi -Wextra -Wno-deprecated -std=c++17 -I. -I../../..
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
$(shell touch dummyIncl/filesystem)
$(shell touch dummyIncl/fstream)
$(shell touch dummyIncl/iomanip)
$(shell touch dummyIncl/iostream)
$(shell touch dummyIncl/limits)
$(shell touch dummyIncl/map)
$(shell touch dummyIncl/memory)
$(shell touch dummyIncl/set)
$(shell touch dummyIncl/sstream)
$(shell touch dummyIncl/string)
$(shell touch dummyIncl/tuple)
$(shell touch dummyIncl/vector)
$(shell touch dummyIncl/unordered_map)
$(shell touch dummyIncl/unordered_set)
$(shell touch dummyIncl/utility)
$(shell touch dummyIncl/algorithm)
$(shell touch dummyIncl/ext/hash_set)
$(shell touch dummyIncl/sys/time.h)
$(shell touch dummyIncl/sys/stat.h)
$(shell touch dummyIncl/sys/types.h)
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

# reduceRange for standalone
ifeq ("$(wildcard DataFormats/Math/interface)","")
$(shell mkdir DataFormats)
$(shell mkdir DataFormats/Math)
$(shell mkdir DataFormats/Math/interface)
endif

ifeq ("$(wildcard DataFormats/Math/interface/deltaPhi.h)","")
$(shell echo "#ifndef DataFormats_Math_interface_deltaPhi_h" > DataFormats/Math/interface/deltaPhi.h)
$(shell echo "#define DataFormats_Math_interface_deltaPhi_h" >> DataFormats/Math/interface/deltaPhi.h)
$(shell echo "namespace reco {" >> DataFormats/Math/interface/deltaPhi.h)
$(shell echo "  inline double reduceRange(double phi) { assert(std::abs(phi) < 100.0); " >> DataFormats/Math/interface/deltaPhi.h)
$(shell echo "    while (phi < -M_PI) { phi += 2 * M_PI; }" >> DataFormats/Math/interface/deltaPhi.h)
$(shell echo "    while (phi > M_PI) { phi -= 2 * M_PI; }" >> DataFormats/Math/interface/deltaPhi.h)
$(shell echo "    return phi; } " >> DataFormats/Math/interface/deltaPhi.h)
$(shell echo "};" >> DataFormats/Math/interface/deltaPhi.h)
$(shell echo "namespace angle0to2pi {" >> DataFormats/Math/interface/deltaPhi.h)
$(shell echo " template <class valType> " >> DataFormats/Math/interface/deltaPhi.h)
$(shell echo " inline constexpr valType make0To2pi(valType angle) {" >> DataFormats/Math/interface/deltaPhi.h)
$(shell echo "   constexpr valType twoPi = 2*M_PI;" >> DataFormats/Math/interface/deltaPhi.h)
$(shell echo "   constexpr valType oneOverTwoPi = 1. / twoPi;" >> DataFormats/Math/interface/deltaPhi.h)
$(shell echo "  constexpr valType epsilon = 1.e-13;" >> DataFormats/Math/interface/deltaPhi.h)
$(shell echo "  if ((std::abs(angle) <= epsilon) || (std::abs(twoPi - std::abs(angle)) <= epsilon))" >> DataFormats/Math/interface/deltaPhi.h)
$(shell echo "    return (0.);" >> DataFormats/Math/interface/deltaPhi.h)
$(shell echo "  if (std::abs(angle) > twoPi) { " >> DataFormats/Math/interface/deltaPhi.h)
$(shell echo "    valType nFac = trunc(angle * oneOverTwoPi);" >> DataFormats/Math/interface/deltaPhi.h)
$(shell echo "    angle -= (nFac * twoPi);" >> DataFormats/Math/interface/deltaPhi.h)
$(shell echo "    if (std::abs(angle) <= epsilon)" >> DataFormats/Math/interface/deltaPhi.h)
$(shell echo "	    return (0.);" >> DataFormats/Math/interface/deltaPhi.h)
$(shell echo "    } " >> DataFormats/Math/interface/deltaPhi.h)
$(shell echo "    if (angle < 0.)" >> DataFormats/Math/interface/deltaPhi.h)
$(shell echo "      angle += twoPi;" >> DataFormats/Math/interface/deltaPhi.h)
$(shell echo "    return (angle);" >> DataFormats/Math/interface/deltaPhi.h)
$(shell echo "  }" >> DataFormats/Math/interface/deltaPhi.h)
$(shell echo "};" >> DataFormats/Math/interface/deltaPhi.h)
$(shell echo "#endif" >> DataFormats/Math/interface/deltaPhi.h)
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

CXXFLAGS += -std=c++17 #fix, otherwise ROOT cflags overrule c++11

LIBS = -lHtml

ifneq ($(CMSSW_BASE),)
	ROOTFIT_INCLUDE := $(shell cd $(CMSSW_BASE); scram tool info roofitcore | grep INCLUDE= | sed 's|INCLUDE=||')
	ROOTFIT_LIBDIR := $(shell cd $(CMSSW_BASE); scram tool info roofitcore | grep LIBDIR= | sed 's|LIBDIR=||')

	CINTINCLUDES := -I$(ROOTFIT_INCLUDE)
ifneq ($(ROOTFIT_INCLUDE),)
	CXXFLAGS     += -I$(ROOTFIT_INCLUDE)
endif
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
	@rm -f fpga
	@rm -f fpga.a
	@rm -rf DataFormats/
	@rm -rf FWCore/
	@rm -rf lib/
	@rm -rf dummyIncl/
	@rm -rf .deps/
	@rm -f *.pdf *.png
	@rm -f allprojections.txt matchcalculator.txt matcheff.txt matchengine.txt matchprocessor.txt trackletcalculator.txt trackletengine.txt trackletprocessor.txt


