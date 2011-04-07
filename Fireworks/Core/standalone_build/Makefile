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
		COMPILERFLAGS := -Wall -O2 -m32 -fPIC ${USER_CXXFLAGS} #-lprofiler
		SO_EXT := so
		MAKESHAREDLIB := -shared
else
	ifeq ($(shell uname), Darwin)
		CC := c++-4.0
		COMPILERFLAGS := -isysroot /Developer/SDKs/MacOSX10.5.sdk -mmacosx-version-min=10.5 -arch i386 -Wall -O2 -fPIC ${USER_CXXFLAGS}
		SO_EXT := dylib
		MAKESHAREDLIB := -mmacosx-version-min=10.5 -dynamiclib
	endif
endif

export CC
export COMPILERFLAGS
export SO_EXT
export MAKESHAREDLIB

# linker
export LINKER = $(CC)

DIR := $(shell pwd)
export FWROOTSYS = ${DIR}/external/root
export LDLIBRARYPATH := `pwd` `pwd`/external/lib $(FWROOTSYS)/lib

export LinkerOptions := -Wl,-rpath -Wl,

# linux specific, tarball ships compiler
ifeq ($(shell uname), Linux)
	export LD_LIBRARY_PATH := /usr/local/lib:${DIR}/external/gcc/lib:/usr/lib:$(FWROOTSYS)/lib
	export PATH :=	${DIR}/external/gcc/bin:/Usr/bin:$(PATH)
else
	export PATH :=	${DIR}/external/gcc/bin:$(PATH)
endif
##############################################################################

#############################################################################

all: display

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
	rm -rf tmp/src; rm libCMSShow.$(SO_EXT); \
	if [ $(shell uname) == "Darwin" ]; then rm -rf lib$(LibProject).$(SO_EXT).dSYM; fi; \
	echo "done"
