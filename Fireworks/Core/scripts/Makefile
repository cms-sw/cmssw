# -*-make-*-
#  Autors:   Dmytro Kovalskyi
#            Johannes Muelmenstaedt
# 
ifndef VERBOSE
  QUIET := @
endif

include core.mk
include project.mk

LinkerOptions := -Wl,-rpath -Wl,
# 
# common compiler flags
# COMPILERFLAGS := -Wall -O2
COMPILERFLAGS := -Wall -O0 -g
# common includes
INCLUDE := -I./ -Isrc -I$(ROOTSYS)/include -Icms $(addprefix -I,$(CoreIncludes)) $(addprefix -I,$(ProjectIncludes))
# compiler
CC := c++

CFLAGS := $(COMPILERFLAGS) $(INCLUDE)  $(EXTRACFLAGS)

# linker
LINKER = c++
# linker flags
# I guessed the following includes. There should be a better way to get them
# -lMatrix -lEG -lGraf -lHist -lGraf3d -lGpad
CoreMissingFlags = -lEG
CoreLinkerFlags = -shared -Lexternal/lib -L$(ROOTSYS)/lib -Lexternal/lib $(addprefix -l,$(CoreLibs)) $(CoreMissingFlags)

ProjectMissingFlags = -lGeom
ProjectLinkerFlags = -shared -Lexternal/lib -L$(ROOTSYS)/lib $(addprefix -l,$(ProjectLibs)) $(ProjectMissingFlags)

# cms data format source files and their object files
CoreSources = $(wildcard cms/*/*/src/*.cc)
CoreObjects = $(addprefix tmp/,$(CoreSources:.cc=.o))
CoreSourcesC = $(wildcard cms/*/*/src/*.c)
CoreObjectsC = $(addprefix tmp/,$(CoreSourcesC:.c=.co))
# CoreObjects = $(addprefix tmp/,$(addsuffix .o,$(basename $(CoreSources)Ñ )))
# here come dependencies
# RequiredCoreObjects = $(CoreObjects)

# cmsShow source code and its object files
ProjectSources = $(wildcard src/*/*/src/*.cc)
ProjectBuildFiles = $(wildcard src/*/*/BuildFile)
ProjectObjects = $(addprefix tmp/,$(ProjectSources:.cc=.o))

# cms data format dictionaries
CoreRootDicHeaders = $(wildcard cms/*/*/src/*LinkDef.h)
CoreRootDicSources = $(addprefix tmp/,$(CoreRootDicHeaders:.h=.cc))
CoreRootDicObjects = $(CoreRootDicSources:.cc=.ro)

CoreDictionaryIncludes = $(wildcard cms/*/*/src/classes.h)
CoreDictionaryXMLs     = $(CoreDictionaryIncludes:.h=_def.xml)
CoreDictionarySources  = $(addprefix tmp/,$(CoreDictionaryIncludes:.h=.cpp))
CoreDictionaryObjects  = $(CoreDictionarySources:.cpp=.do)

# cmsShow dictionaries
ProjectRootDicHeaders = $(wildcard src/*/*/src/*LinkDef.h)
ProjectRootDicSources = $(addprefix tmp/,$(ProjectRootDicHeaders:.h=.cc))
ProjectRootDicObjects = $(ProjectRootDicSources:.cc=.ro)

ProjectDictionaryIncludes = $(wildcard src/*/*/src/classes.h)
ProjectDictionaryXMLs     = $(ProjectDictionaryIncludes:.h=_def.xml)
ProjectDictionarySources  = $(addprefix tmp/,$(ProjectDictionaryIncludes:.h=.cpp))
ProjectDictionaryObjects  = $(ProjectDictionarySources:.cpp=.do)

LibCore = CMSDataFormats
LibProject = CMSShow
# ProjectLibs = Tree FTGL Gpad Eve Physics Graf3d Ged Geom Gui RGL Graf RIO TreePlayer Rint GeomPainter Hist EG

LDLIBRARYPATH := `pwd` `pwd`/external/lib $(ROOTSYS)/lib

all: core display
	$(QUIET) echo '#!/bin/sh' > root.sh; \
	echo "CMSSW_BASE="`pwd` >> root.sh; \
	echo "export CMSSW_BASE" >> root.sh; \
	echo "ROOTSYS=$(ROOTSYS)" >> root.sh; \
	echo "export ROOTSYS" >> root.sh; \
	echo "LD_LIBRARY_PATH=$(LDLIBRARYPATH)" | perl -ne 's/ +/:/g;print' >> root.sh; \
	echo "export LD_LIBRARY_PATH" >> root.sh; \
	echo "$(ROOTSYS)/bin/root.exe" '$$*' >> root.sh; \
	chmod u+x root.sh

core:   env lib$(LibCore).so tmp/lib$(LibCore).so.out
	
display: env lib$(LibProject).so tmp/lib$(LibProject).so.out

# check if proper environment is setup
env: 
	$(QUIET) if [ -z "$(ROOTSYS)" ]; then \
	   echo "ROOTSYS not set. Cannot compile dictionaries."; \
	   exit 1; \
	fi;

lib$(LibProject).so: $(ProjectObjects) $(ProjectRootDicObjects) $(ProjectDictionaryObjects) lib$(LibCore).so
	$(QUIET) echo "linking core library $(LibProject)"; \
	$(LINKER) $(ProjectLinkerFlags) -L./ -l$(LibCore) -o lib$(LibProject).so \
	$(ProjectObjects) $(ProjectRootDicObjects) $(ProjectDictionaryObjects)

lib$(LibCore).so: $(CoreObjects) $(CoreObjectsC) $(CoreRootDicObjects) $(CoreDictionaryObjects)
	$(QUIET) echo "linking core library $(LibCore)"; \
	$(LINKER) $(CoreLinkerFlags) -o lib$(LibCore).so \
	$(CoreObjects) $(CoreObjectsC) $(CoreRootDicObjects) $(CoreDictionaryObjects)

######################################################################################
# 
#                General rules for making object files
# 
######################################################################################
# get object files
tmp/%.o:   %.cc
	$(QUIET) echo "compiling $<"; \
	mkdir -p $(dir $@); \
	$(CC) $(CFLAGS) $< -c -o $@

# get object files for C code
tmp/%.co:   %.c
	$(QUIET) echo "compiling $<"; \
	mkdir -p $(dir $@); \
	$(CC) $(CFLAGS) $< -c -o $@

# dictionary creation
tmp/%.cpp:  %.h %_def.xml
	$(QUIET) echo "generating dictionaries based on $*_def.xml"; \
	mkdir -p $(dir $@); \
	$(ROOTSYS)/bin/genreflex $*.h -s $*_def.xml -o $@ $(INCLUDE) --gccxmlpath=external/gccxml/bin

# dictionary creation
# NOTE: for some reason I needed to add the original
#   LinkDef file contect to the generated dictionary code to be 
#   able to compile
tmp/%LinkDef.cc:  %LinkDef.h
	$(QUIET) echo "generating ROOT dictionaries based on $<"; \
	mkdir -p $(dir $@); \
	LD_LIBRARY_PATH=$(ROOTSYS)/lib; export LD_LIBRARY_PATH; \
	$(ROOTSYS)/bin/rootcint -f $@.tmp -c -p $(INCLUDE) $<; \
	cat $< $@.tmp > $@

# object files for dictionaries
%.do:   %.cpp
	$(QUIET) echo "compiling dictionaries $<"; \
	$(CC) $(CFLAGS) -I$(ROOTSYS)/include $< -c -o $@

# object files for dictionaries
%.ro:   %.cc
	$(QUIET) echo "compiling ROOT dictionaries $<"; \
	$(CC) $(CFLAGS) $(INCLUDE) $< -c -o $@

# check dependencies
# -lRIO -lNet - "hacks" for standard ROOT to be able to use the
# recipe below to check for undefined symbols
tmp/%.out:  %
	$(QUIET) echo "checking shared library for missing symbols: $<"; \
	echo "int main(){}" > tmp/$<.cpp; \
	$(CC) $(CFLAGS) -Wl,-rpath -Wl,./ -L$(ROOTSYS)/lib $(addprefix $(LinkerOptions),$(LDLIBRARYPATH)) -lRIO -lNet -o $@ $< tmp/$<.cpp

clean: cleancore cleandisplay

cleancore:  
	$(QUIET) echo "removing temporary core objects and output library..."; \
	rm -rf tmp/cms lib$(LibCore).so; echo "done"
cleandisplay:  
	$(QUIET) echo "removing temporary objects and output libraries..."; \
	rm -rf tmp/src lib$(LibProject).so; echo "done"
project.mk: extractCMSDataFormats.pl $(ProjectBuildFiles)
	$(QUIET) ./extractCMSDataFormats.pl --mode project --dir ./
