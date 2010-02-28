# -*-make-*-
#  Autors:   Dmytro Kovalskyi
#            Johannes Muelmenstaedt
# 

# debug
ifndef VERBOSE
  export QUIET := @
endif

# library extensions
ifeq ($(shell uname), Linux)
		CC := c++
		COMPILERFLAGS := -Wall -O2 -m32 -fPIC
		SO_EXT := so
		MAKESHAREDLIB := -shared
		LinkerOptions := -Wl,-rpath -Wl,
else 
	ifeq ($(shell uname), Darwin)
		CC := c++-4.0
		COMPILERFLAGS := -V 4.0.1 -isysroot /Developer/SDKs/MacOSX10.5.sdk -mmacosx-version-min=10.5 -arch i386 -x c++ -m32 -Wall -O2 -fPIC
		SO_EXT := dylib
		MAKESHAREDLIB := -dynamiclib
		LinkerOptions := -Wl,-rpath -Wl,
	endif
endif

export CC
export COMPILERFLAGS
export LinkerOptions
export SO_EXT
export MAKESHAREDLIB

# linker
export LINKER = $(CC)

DIR := $(shell pwd)
export FWROOTSYS = ${DIR}/external/root
export LDLIBRARYPATH := `pwd` `pwd`/external/lib $(FWROOTSYS)/lib

# linux specific, tarball ships compiler 
export LD_LIBRARY_PATH := ${DIR}/external/gcc/lib:/usr/lib:$(FWROOTSYS)/lib
export PATH :=	${DIR}/external/gcc/bin:/usr/bin:$(PATH)

##############################################################################

#############################################################################

all: core display

core:   Makefile.core
	$(QUIET) $(MAKE) -f Makefile.core

display: Makefile.project
	$(QUIET) $(MAKE) -f Makefile.project

clean: cleandisplay

cleanall: cleancore cleandisplay

cleancore: 
	$(QUIET) echo "removing temporary core objects and output library..."; \
	rm -rf tmp/cms; rm libCMSDataFormats.$(SO_EXT); echo "done"

cleandisplay: 
	$(QUIET) echo "removing temporary project objects and output libraries..."; \
	rm -rf tmp/src; rm libCMSShow.$(SO_EXT); echo "done"

