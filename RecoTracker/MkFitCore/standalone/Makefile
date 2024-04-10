include ${SADIR}/Makefile.config

CPPFLAGS := -I. -I${SRCDIR} -I../mkFit-external ${CPPFLAGS}

CORE_DIR := ${SRCDIR}/RecoTracker/MkFitCore

LIB_CORE := ../libMicCore.so

TGTS := ${LIB_CORE}

.PHONY: all clean distclean


SRCS := $(wildcard ${CORE_DIR}/src/*.cc) \
		$(wildcard ${CORE_DIR}/src/Matriplex/*.cc) \
		$(wildcard ${SADIR}/*.cc)
SRCB := $(notdir ${SRCS})
DEPS := $(SRCB:.cc=.d)
OBJS := $(SRCB:.cc=.o)

vpath %.cc ${CORE_DIR}/src ${CORE_DIR}/src/Matriplex ${SADIR}

AUTO_TGTS :=

# Begin ROOT dumpers
# XXX can remove, is var empty if undefined ????
RNTD_OBJS :=

ifdef WITH_ROOT

LIB_RNTDUMP := ../libMicRntDump.so
RNTDUMP_DICT_PCM := ../MicRntDumpDict_rdict.pcm

RNTD_SRCS := $(wildcard ${SADIR}/RntDumper/*.cc)
RNTD_SRCB := $(notdir ${RNTD_SRCS})
RNTD_OBJS := $(RNTD_SRCB:.cc=.o)
DEPS += $(RNTD_SRCB:.cc=.d)
TGTS += ${LIB_RNTDUMP}

#LIB_CORE_DEPS := ${LIB_RNTDUMP}
#LIB_CORE_LD_EXTRA := -L.. -lMicRntDump

vpath %.cc ${SADIR}/RntDumper

endif
# End ROOT dumpers

all: ${TGTS}

# Begin Matriplex

auto-matriplex:
### Matriplex auto-generated fragments are included in source.
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
	-rm -rf MicRntDumpDict.cc auto-matriplex main.dSYM
	-rm -rf plotting/*.so plotting/*.d plotting/*.pcm

clean: clean-local

distclean: clean-local
	-rm -f ${AUTO_TGTS}
	-rm -f *.optrpt
	-rm -f ${LIB_CORE}
###	cd Matriplex && ${MAKE} distclean

${LIB_CORE}: ${OBJS} ${LIB_RNTDUMP}
	@mkdir -p $(@D)
	${CXX} ${CXXFLAGS} ${VEC_HOST} ${OBJS} -shared -o $@ ${LDFLAGS_HOST} ${LDFLAGS} ${LIB_CORE_LD_EXTRA}

ifdef WITH_ROOT
${LIB_RNTDUMP}: ${RNTD_OBJS} MicRntDumpDict.o
	${CXX} ${CXXFLAGS} ${VEC_HOST} ${RNTD_OBJS} MicRntDumpDict.o -shared -o $@ ${LDFLAGS_HOST} ${LDFLAGS} -lROOTEve

MicRntDumpDict.cc ${RNTDUMP_DICT_PCM} &: ${SADIR}/RntDumper/RntStructs.h ${SADIR}/RntDumper/RntStructs_Linkdef.h
	rootcling -I. -I${SRCDIR} -f MicRntDumpDict.cc ${SADIR}/RntDumper/RntStructs.h ${SADIR}/RntDumper/RntStructs_Linkdef.h
	mv MicRntDumpDict_rdict.pcm ${RNTDUMP_DICT_PCM}
endif

${OBJS} ${RNTD_OBJS}: %.o: %.cc %.d
	${CXX} ${CPPFLAGS} ${CXXFLAGS} ${VEC_HOST} -c -o $@ $<

%.s: %.cc
	${CXX} ${CPPFLAGS} ${CXXFLAGS} ${VEC_HOST} -S -o $@ $<

%.d: %.cc
	${MAKEDEPEND} -o $@ $<

echo:
	@echo "CXX=${CXX}"
	@echo SRCS = ${SRCS}
	@echo DEPS = ${DEPS}
	@echo OBJS = ${OBJS}
	@echo RNTD_OBJS = ${RNTD_OBJS}

echo_cc_defs:
	${CXX} -dM -E -mavx2 - < /dev/null

echo-srcs:
	@echo ${SRCS}

echo-flags:
	@echo "CPPFLAGS=${CPPFLAGS}"

echo-tbb:
	@echo "TBB_GCC=${TBB_GCC}, TBB_PREFIX=${TBB_PREFIX}, TBB_ROOT=${TBB_ROOT}"
