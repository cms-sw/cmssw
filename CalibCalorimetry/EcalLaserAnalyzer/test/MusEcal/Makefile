#
# Makefile for MusEcal
#
# To link with other packages, add:
#
# CXXFLAGS   += -I${ANOTHERPKG}/src ... 
# ROOTGLIBS  += -lRooFitCore -lRooFit
# ROOTGLIBS  += -L${ANOTHERPKG}/lib -lANOTHERPKG
# ...

PKGNAME := MusEcal

MusEcal := .

### global options

# Architecture
ARCH         := $(shell root-config --arch)

# File suffixes
OBJSUFFIX  := o
DEPSUFFIX  := d
DLLSUFFIX  := so

# File directories
OBJDIR     := .o
DEPDIR     := .d
BINDIR     := $(MusEcal)/bin

# output directories
SOLIBDIR   := $(MusEcal)/lib
SOLIB      := $(SOLIBDIR)/lib$(PKGNAME).$(DLLSUFFIX)

__dummy := $(shell mkdir -p $(OBJDIR) $(DEPDIR) $(BINDIR) $(SOLIBDIR))

### compilator, linker, flags and options
CXX        := g++
LD         := g++
OUTPUTOPT  := -o
OPTFLAGS   := -g -O2 -Wall -fPIC -Wno-deprecated
CXXFLAGS   := $(OPTFLAGS)
LDFLAGS    := -shared
LIBS       :=

# Architecture dependant options
-include $(ARCH).mk


### processing of the modules
# find the modules to include and compile
MODULES    := $(subst /module.mk,,$(wildcard */module.mk))

__dummy := $(shell ./genmak.sh $(MODULES))

all: lib prog

# the source files to compile (without suffix)
SRC        :=
PROG       :=
PROGSRC    :=
#VPATH      :=

-include gen.mk

### build targets and include path for compilation
OBJS       := $(addprefix ${OBJDIR}/, $(addsuffix .$(OBJSUFFIX), $(SRC)))
DEPS       := $(addprefix ${DEPDIR}/, $(addsuffix .$(DEPSUFFIX), $(SRC)))
DEPS       += $(addprefix ${DEPDIR}/, $(addsuffix .$(DEPSUFFIX), $(PROGSRC)))
CXXFLAGS   += -I. -I.. -I${MECore}/src $(patsubst %, -I%, $(MODULES))

### External libraries
# ROOT
ROOTCFLAGS := $(shell root-config --cflags)
# add CMSSW library path to link to proper libraries...
CMSSW_LIBS  = -L$(subst :, -L, $(LD_LIBRARY_PATH) ) -L/lib
CMSSW_LIBS_PARSED= $(subst ;,, $(CMSSW_LIBS) )
ROOTGLIBS  := $(CMSSW_LIBS_PARSED)
ROOTGLIBS  += $(shell root-config --glibs) -lMinuit -lHtml -lEG -lThread
#ROOTGLIBS  += -L${MECore}/lib -lMECore
ROOTGLIBS  += -lCalibCalorimetryEcalLaserAnalyzer

# add possible new dependencies
# -lMinuit2
#ROOTGLIBS  += -L../HAF/lib -lHAF
#ROOTGLIBS  += -L../DataFormats/lib -lCandidate

ROOTCINT   := rootcint
CXXFLAGS   += $(ROOTCFLAGS)
LIBS       += $(ROOTGLIBS) 

# libxml2
#CXXFLAGS   += $(shell xml2-config --cflags)
#LIBS       += $(shell xml2-config --libs)

### Rules
.PHONY: all clean

.PRECIOUS: ${OBJDIR}/%.bin %SK.cc %.hh

lib: $(SOLIB)

prog: $(PROG)

#gen.mk dictgen.mk: $(MODULES)
#	./genmak.sh $(MODULES)

# the main library rule
$(SOLIB): $(OBJS)
	@echo "Linking $^"
	@$(LD) $(LDFLAGS) $^ $(LIBS) $(FLIBS) $(OUTPUTOPT) $@
	@echo "Successfully completed $@"

-include dictgen.mk

cleanAll:
	@echo "Cleaning up..."
	@echo $(DEPS)
	@echo $(OBJS)
	@echo $(SOLIB)
	@echo $(PROG)
	@echo $(PROGSRC)

# clean rule
clean:
	@echo "Cleaning up..."
	@echo $(DEPS)
	@echo $(OBJS)
	@echo $(SOLIB)
	@rm -f $(DEPS) $(OBJS) $(SOLIB) $(foreach i,$(MODULES),$(i)/dict_$(i).*) gen.mk dictgen.mk

######

# .o building rules
${OBJDIR}/%.$(OBJSUFFIX) : %.cxx ${DEPDIR}/%.$(DEPSUFFIX)
	@echo "Compiling $<"
	@$(CXX) $(CXXFLAGS) -c $< -o $@

${OBJDIR}/%.$(OBJSUFFIX) : %.cpp ${DEPDIR}/%.$(DEPSUFFIX)
	@echo "Compiling $<"
	@$(CXX) $(CXXFLAGS) -c $< -o $@

${OBJDIR}/%.$(OBJSUFFIX) : %.cc ${DEPDIR}/%.$(DEPSUFFIX)
	@echo "Compiling $<"
	@$(CXX) $(CXXFLAGS) -c $< -o $@

# .d building rules
${DEPDIR}/%.$(DEPSUFFIX): %.cxx
#	@echo "Making $@"
	@$(CXX) -MM $(CXXFLAGS) $< \
	| sed 's|\('`basename $*`'\)\.o[ :]*|'$(OBJDIR)'/'`dirname $*`'/\1.o '$(DEPDIR)'/'`dirname $*`'/\1.d : |g' \
	> $@

${DEPDIR}/%.$(DEPSUFFIX): %.cpp
#	@echo "Making $@"
	@$(CXX) -MM $(CXXFLAGS) $< \
	| sed 's|\('`basename $*`'\)\.o[ :]*|'$(OBJDIR)'/'`dirname $*`'/\1.o '$(DEPDIR)'/'`dirname $*`'/\1.d : |g' \
	> $@

${DEPDIR}/%.$(DEPSUFFIX): %.cc
#	@echo "Making $@"
	@$(CXX) -MM $(CXXFLAGS) $< \
	| sed 's|\('`basename $*`'\)\.o[ :]*|'$(OBJDIR)'/'`dirname $*`'/\1.o '$(DEPDIR)'/'`dirname $*`'/\1.d : |g' \
	> $@

# .bin building rules
${OBJDIR}/%.bin: ${OBJDIR}/%.$(OBJSUFFIX) | $(SOLIB)
	@echo "Making $<"
	@$(CXX) $(CXXFLAGS) $(LIBS) -L$(SOLIBDIR) -l$(PKGNAME) $^ -o $@

ifneq ($(MAKECMDGOALS),clean)
-include $(DEPS)
endif

