include ${SADIR}/Makefile.config

CPPFLAGS := -I${SRCDIR} -I../mkFit-external ${CPPFLAGS}

.PHONY: all clean distclean echo

SRCS := $(wildcard ${SADIR}/Geoms/*.cc)
SRCB := $(notdir ${SRCS})
DEPS := $(SRCB:.cc=.d)
OBJS := $(SRCB:.cc=.o)

TGTS := $(basename ${OBJS})
TGTS := $(addprefix ../, $(addsuffix .so, ${TGTS}))

vpath %.cc ${SADIR}/Geoms

all: ${TGTS}

%.o: %.cc %.d
	${CXX} ${CPPFLAGS} ${CXXFLAGS} ${VEC_HOST} -c -o $@ $<

%.d: %.cc
	${MAKEDEPEND} -o $@ $<

../%.so: %.o
	${CXX} -shared -L.. -lMicCore -o $@ $<

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
