include ${SADIR}/Makefile.config

CPPFLAGS := -I${SRCDIR} -I../mkFit-external ${CPPFLAGS}

.PHONY: all clean distclean echo

SRCS := ${SACMS}/Geoms/CMS-phase1.cc ${SACMS}/Geoms/CMS-phase2.cc
SRCB := $(notdir ${SRCS})
DEPS := $(SRCB:.cc=.d)
OBJS := $(SRCB:.cc=.o)

TGTS := $(basename ${OBJS})
TGTS := $(addprefix ../, $(addsuffix .so, ${TGTS}))

GEO_phase1_BIN = ../CMS-phase1.bin
GEO_phase2_BIN = ../CMS-phase2.bin

vpath %.cc ${SACMS}/Geoms

all: ${TGTS} ${GEO_phase1_BIN} ${GEO_phase2_BIN}

%.o: %.cc %.d
	${CXX} ${CPPFLAGS} ${CXXFLAGS} ${VEC_HOST} -c -o $@ $<

%.d: %.cc
	${MAKEDEPEND} -o $@ $<

../%.so: %.o
	${CXX} -shared -L.. -lMicCore -o $@ $<

${GEO_phase1_BIN}:
	curl http://xrd-cache-1.t2.ucsd.edu/matevz/PKF/CMS-phase1.bin -o $@

${GEO_phase2_BIN}:
	curl http://xrd-cache-1.t2.ucsd.edu/matevz/PKF/CMS-phase2.bin -o $@

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
