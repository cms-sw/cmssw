ifndef ROOTSYS
$(error ROOTSYS is not set. Please set ROOT environment properly)
endif

WITH_ROOT:=1

include ../Makefile.config

.PHONY: all clean distclean echo

all: default

CPPEXTRA := -I.. ${USER_CPPFLAGS} ${DEFS}
LDEXTRA  := ${USER_LDFLAGS}

CPPFLAGS := -I${ROOTSYS}/include ${CPPEXTRA} ${CPPFLAGS}
CXXFLAGS += ${USER_CXXFLAGS}
LDFLAGS  += -L${ROOTSYS}/lib -L../lib -lCore -lRIO -lTree -lMathCore ${LDEXTRA}


EXES     := writeMemoryFile

default: ${EXES}

clean:
	rm -f ${EXES} *.d *.o *.om *.so *Dict.* *.pcm
	rm -rf writeMemoryFile.dSYM

distclean: clean
	rm -f *.optrpt 
	rm -f ${EXES} *.ah

echo:
	@echo "CXX      = ${CXX}"
	@echo "CPPFLAGS = ${CPPFLAGS}"
	@echo "CXXFLAGS = ${CXXFLAGS}"
	@echo "LDFLAGS  = ${LDFLAGS}"
	@echo "EXES     = ${EXES}"


################################################################

SRCS := WriteMemoryFile.cc

OBJS := $(SRCS:.cc=.o)

#libDicts.so: DictsDict.o
#	${CXX} ${CXXFLAGS} ${LDFLAGS} $< -shared -o $@ ${LDFLAGS_HOST}

DictsDict.cc: DictsLinkDef.h
	rootcint -v3 -f $@ -c -p $<

writeMemoryFile: ${OBJS} DictsDict.o
	${CXX} ${CXXFLAGS} ${LDFLAGS} $^ -L.. -lMicCore -Wl,-rpath=..,-rpath=.,-rpath=../lib -o $@

%.o: %.cc
	${CXX} ${CPPFLAGS} ${CXXFLAGS} -c -o $@ $<
