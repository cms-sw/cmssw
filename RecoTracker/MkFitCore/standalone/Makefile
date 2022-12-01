include ${SADIR}/Makefile.config

CPPFLAGS := -I. -I${SRCDIR} -I../mkFit-external ${CPPFLAGS}

CORE_DIR := ${SRCDIR}/RecoTracker/MkFitCore

LIB_CORE := ../libMicCore.so

TGTS := ${LIB_CORE}

.PHONY: all clean distclean

all: ${TGTS}

SRCS := $(wildcard ${CORE_DIR}/src/*.cc) \
		$(wildcard ${CORE_DIR}/src/Matriplex/*.cc) \
		$(wildcard ${SADIR}/*.cc)
SRCB := $(notdir ${SRCS})
DEPS := $(SRCB:.cc=.d)
OBJS := $(SRCB:.cc=.o)

vpath %.cc ${CORE_DIR}/src ${CORE_DIR}/src/Matriplex ${SADIR}

AUTO_TGTS :=

# Begin Matriplex

auto-matriplex:
###	${MAKE} -f ${CORE_DIR}/src/Matriplex auto && touch $@
	touch $@

AUTO_TGTS += auto-matriplex

${DEPS}: auto-matriplex

# End Matriplex

ifeq ($(filter clean-local clean distclean, ${MAKECMDGOALS}),)
include ${DEPS}
endif

clean-local:
	-rm -f ${TGTS} *.d *.o *.om *.so
	-rm -rf main.dSYM
	-rm -rf plotting/*.so plotting/*.d plotting/*.pcm

clean: clean-local

distclean: clean-local
	-rm -f ${AUTO_TGTS}
	-rm -f *.optrpt
	-rm -f ${LIB_CORE}
###	cd Matriplex && ${MAKE} distclean

${LIB_CORE}: ${OBJS}
	@mkdir -p $(@D)
	${CXX} ${CXXFLAGS} ${VEC_HOST} ${OBJS} -shared -o $@ ${LDFLAGS_HOST} ${LDFLAGS}

${OBJS}: %.o: %.cc %.d
	${CXX} ${CPPFLAGS} ${CXXFLAGS} ${VEC_HOST} -c -o $@ $<

%.d: %.cc
	${MAKEDEPEND} -o $@ $<

echo:
	@echo "CXX=${CXX}"
	@echo SRCS = ${SRCS}
	@echo DEPS = ${DEPS}
	@echo OBJS = ${OBJS}

echo_cc_defs:
	${CXX} -dM -E -mavx2 - < /dev/null

echo-srcs:
	@echo ${SRCS}

echo-flags:
	@echo "CPPFLAGS=${CPPFLAGS}"

echo-tbb:
	@echo "TBB_GCC=${TBB_GCC}, TBB_PREFIX=${TBB_PREFIX}, TBB_ROOT=${TBB_ROOT}"
