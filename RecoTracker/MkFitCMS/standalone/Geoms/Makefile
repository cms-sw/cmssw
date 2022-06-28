include ${SADIR}/Makefile.config

CPPFLAGS := -I${SRCDIR} -I../mkFit-external ${CPPFLAGS}

.PHONY: all clean distclean echo

SRCS := ${SACMS}/Geoms/CMS-2017.cc ${SACMS}/Geoms/CMS-2029.cc
SRCB := $(notdir ${SRCS})
DEPS := $(SRCB:.cc=.d)
OBJS := $(SRCB:.cc=.o)

TGTS := $(basename ${OBJS})
TGTS := $(addprefix ../, $(addsuffix .so, ${TGTS}))

GEO_2017_BIN = ../CMS-2017.bin
GEO_2029_BIN = ../CMS-2029.bin

vpath %.cc ${SACMS}/Geoms

all: ${TGTS} ${GEO_2017_BIN} ${GEO_2029_BIN}

%.o: %.cc %.d
	${CXX} ${CPPFLAGS} ${CXXFLAGS} ${VEC_HOST} -c -o $@ $<

%.d: %.cc
	${MAKEDEPEND} -o $@ $<

../%.so: %.o
	${CXX} -shared -L.. -lMicCore -o $@ $<

${GEO_2017_BIN}:
	curl http://xrd-cache-1.t2.ucsd.edu/matevz/PKF/CMS-2017.bin -o $@

${GEO_2029_BIN}:
	curl http://xrd-cache-1.t2.ucsd.edu/matevz/PKF/CMS-2029.bin -o $@

ifeq ($(filter clean distclean, ${MAKECMDGOALS}),)
include ${DEPS}
endif

clean:
	-rm -f *.so *.o *.om *.d *.optrpt

distclean: clean
	-rm -f ${TGTS}

echo:
	@echo SRCS = ${SRCS}
	@echo DEPS = ${DEPS}
	@echo OBJS = ${OBJS}
