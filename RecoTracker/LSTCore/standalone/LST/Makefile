#
# stuff to make
#

CCSOURCES=$(wildcard ../../src/*.cc)
ALPAKACCSOURCES=$(filter-out ../../src.alpaka/LST.dev.cc, $(wildcard ../../src/alpaka/*.dev.cc))
CCOBJECTS_CPU=$(patsubst ../../src/alpaka/%.dev.cc, %_cpu.o, $(ALPAKACCSOURCES)) $(patsubst ../../src/%.cc, %_cpu.o, $(CCSOURCES))
CCOBJECTS_CUDA=$(patsubst ../../src/alpaka/%.dev.cc, %_cuda.o, $(ALPAKACCSOURCES)) $(patsubst ../../src/%.cc, %_cuda.o, $(CCSOURCES))
CCOBJECTS_ROCM=$(patsubst ../../src/alpaka/%.dev.cc, %_rocm.o, $(ALPAKACCSOURCES)) $(patsubst ../../src/%.cc, %_rocm.o, $(CCSOURCES))

LSTSOURCES=../../src/alpaka/LST.dev.cc
LSTOBJECTS_CPU=$(patsubst ../../src/alpaka/%.dev.cc, %_cpu.o, $(LSTSOURCES))
LSTOBJECTS_CUDA=$(patsubst ../../src/alpaka/%.dev.cc, %_cuda.o, $(LSTSOURCES))
LSTOBJECTS_ROCM=$(patsubst ../../src/alpaka/%.dev.cc, %_rocm.o, $(LSTSOURCES))

# Default to CPU and CUDA backends
ifeq ($(BACKEND),)
  LIB_CPU=liblst_cpu.so
  LIB_CUDA=liblst_cuda.so
endif

ifneq ($(findstring cpu,$(BACKEND)),)
  LIB_CPU=liblst_cpu.so
endif
ifneq ($(findstring cuda,$(BACKEND)),)
  LIB_CUDA=liblst_cuda.so
endif
ifneq ($(findstring rocm,$(BACKEND)),)
  LIB_ROCM=liblst_rocm.so
endif
ifneq ($(findstring all,$(BACKEND)),)
  LIB_CPU=liblst_cpu.so
  LIB_CUDA=liblst_cuda.so
  LIB_ROCM=liblst_rocm.so
endif

LIBS=$(LIB_CPU) $(LIB_CUDA) $(LIB_ROCM)

#
# flags to keep track of
#

# Different architectures to optimize for
GENCODE_CUDA := -gencode arch=compute_70,code=[sm_70,compute_70] -gencode arch=compute_89,code=[sm_89,compute_89]

CXX                  = g++
CXXFLAGS_CPU         = -march=native -mtune=native -Ofast -fno-reciprocal-math -fopenmp-simd -g -Wall -Wshadow -Woverloaded-virtual -fPIC -fopenmp -I..
CXXFLAGS_CUDA        = -O3 -g --compiler-options -Wall --compiler-options -Wshadow --compiler-options -Woverloaded-virtual --compiler-options -fPIC --compiler-options -fopenmp -dc -lineinfo --ptxas-options=-v --cudart shared $(GENCODE_CUDA) --use_fast_math --default-stream per-thread -I..
CXXFLAGS_ROCM        = -O3 -g -Wall -Wshadow -Woverloaded-virtual -fPIC -I${ROCM_ROOT}/include -I..
CMSSWINCLUDE        := -I${TRACKLOOPERDIR}/../../../ -I${CMSSW_BASE}/src
ifdef CMSSW_RELEASE_BASE
CMSSWINCLUDE        := ${CMSSWINCLUDE} -I${CMSSW_RELEASE_BASE}/src
endif
ALPAKAINCLUDE        = -I${ALPAKA_ROOT}/include -I/${BOOST_ROOT}/include -std=c++17 ${CMSSWINCLUDE}
ALPAKASERIAL         = -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
ALPAKACUDA           = -DALPAKA_ACC_GPU_CUDA_ENABLED -DALPAKA_ACC_GPU_CUDA_ONLY --expt-relaxed-constexpr
ALPAKAROCM           = -DALPAKA_ACC_GPU_HIP_ENABLED -DALPAKA_ACC_GPU_HIP_ONLY -DALPAKA_DISABLE_VENDOR_RNG
ROOTINCLUDE          = -I$(ROOT_ROOT)/include
ROOTCFLAGS           = -pthread -m64 $(ROOTINCLUDE)
PRINTFLAG            = -DT4FromT3
DUPLICATES           = -DDUP_pLS -DDUP_T5 -DDUP_pT5 -DDUP_pT3 -DCrossclean_T5 -DCrossclean_pT3 #-DFP16_Base
CACHEFLAG            =
PTCUTFLAG            =
LSTWARNINGSFLAG      =
CMSSW_WERRORS_CPU    = -Werror=pointer-arith -Werror=overlength-strings -Werror=return-type -Werror=missing-braces -Werror=unused-value -Werror=unused-label \
                       -Werror=address -Werror=format -Werror=sign-compare -Werror=write-strings -Werror=delete-non-virtual-dtor -Werror=strict-aliasing -Werror=narrowing \
                       -Werror=unused-but-set-variable -Werror=reorder -Werror=unused-variable -Werror=conversion-null -Werror=return-local-addr -Wnon-virtual-dtor -Werror=switch \
                       -Werror=main -Werror=overflow -Werror=format-contains-nul -Werror=type-limits -Wreturn-type -Wextra -Wpessimizing-move -Wclass-memaccess -Wunused \
                       -Wparentheses -Wno-vla -Wno-non-template-friend -Wno-long-long -Wno-cast-function-type -Wno-unused-but-set-parameter -Wno-ignored-qualifiers \
                       -Wno-unused-parameter -Wno-unused-local-typedefs -Wno-attributes
CMSSW_WERRORS_CUDA   = $(patsubst %,-Xcompiler %,$(CMSSW_WERRORS_CPU))
CMSSW_WERRORS_ROCM   = $(CMSSW_WERRORS_CPU)
CACHEFLAG_FLAGS      = -DCACHE_ALLOC
T5CUTFLAGS           = $(T5DNNFLAG) $(T5RZCHI2FLAG) $(T5RPHICHI2FLAG)

LD_CPU               = g++
SOFLAGS_CPU          = -g -shared -fPIC
ALPAKABACKEND_CPU    = $(ALPAKASERIAL)
COMPILE_CMD_CPU      = $(LD_CPU) -c

LD_CUDA              = nvcc
SOFLAGS_CUDA         = -g -shared --compiler-options -fPIC --cudart shared $(GENCODE_CUDA)
ALPAKABACKEND_CUDA   = $(ALPAKACUDA)
COMPILE_CMD_CUDA     = $(LD_CUDA) -x cu

LD_ROCM              = hipcc
SOFLAGS_ROCM         = -g -shared -fPIC
ALPAKABACKEND_ROCM   = $(ALPAKAROCM)
COMPILE_CMD_ROCM     = $(LD_ROCM) -c

CUTVALUEFLAG =
CUTVALUEFLAG_FLAGS = -DCUT_VALUE_DEBUG

%_cpu.o: ../../src/alpaka/%.dev.cc
	$(COMPILE_CMD_CPU) $(CXXFLAGS_CPU) $(ROOTINCLUDE) $(PRINTFLAG) $(CACHEFLAG) $(CUTVALUEFLAG) $(LSTWARNINGSFLAG) $(CMSSW_WERRORS_CPU) $(T5CUTFLAGS) $(PTCUTFLAG) $(DUPLICATES) $(ALPAKAINCLUDE) $(ALPAKABACKEND_CPU) $< -o $@

%_cuda.o: ../../src/alpaka/%.dev.cc
	$(COMPILE_CMD_CUDA) $(CXXFLAGS_CUDA) $(ROOTINCLUDE) $(PRINTFLAG) $(CACHEFLAG) $(CUTVALUEFLAG) $(LSTWARNINGSFLAG) $(CMSSW_WERRORS_CUDA) $(T5CUTFLAGS) $(PTCUTFLAG) $(DUPLICATES) $(ALPAKAINCLUDE) $(ALPAKABACKEND_CUDA) $< -o $@

%_rocm.o: ../../src/alpaka/%.dev.cc
	$(COMPILE_CMD_ROCM) $(CXXFLAGS_ROCM) $(ROOTINCLUDE) $(PRINTFLAG) $(CACHEFLAG) $(CUTVALUEFLAG) $(LSTWARNINGSFLAG) $(CMSSW_WERRORS_ROCM) $(T5CUTFLAGS) $(PTCUTFLAG) $(DUPLICATES) $(ALPAKAINCLUDE) $(ALPAKABACKEND_ROCM) $< -o $@

%_cpu.o: ../../src/%.cc
	$(COMPILE_CMD_CPU) $(CXXFLAGS_CPU) $(ROOTINCLUDE) $(PRINTFLAG) $(CACHEFLAG) $(CUTVALUEFLAG) $(LSTWARNINGSFLAG) $(CMSSW_WERRORS_CPU) $(T5CUTFLAGS) $(PTCUTFLAG) $(DUPLICATES) $(ALPAKAINCLUDE) $(ALPAKABACKEND_CPU) $< -o $@

%_cuda.o: ../../src/%.cc
	$(COMPILE_CMD_CUDA) $(CXXFLAGS_CUDA) $(ROOTINCLUDE) $(PRINTFLAG) $(CACHEFLAG) $(CUTVALUEFLAG) $(LSTWARNINGSFLAG) $(CMSSW_WERRORS_CUDA) $(T5CUTFLAGS) $(PTCUTFLAG) $(DUPLICATES) $(ALPAKAINCLUDE) $(ALPAKABACKEND_CUDA) $< -o $@

%_rocm.o: ../../src/%.cc
	$(COMPILE_CMD_ROCM) $(CXXFLAGS_ROCM) $(ROOTINCLUDE) $(PRINTFLAG) $(CACHEFLAG) $(CUTVALUEFLAG) $(LSTWARNINGSFLAG) $(CMSSW_WERRORS_ROCM) $(T5CUTFLAGS) $(PTCUTFLAG) $(DUPLICATES) $(ALPAKAINCLUDE) $(ALPAKABACKEND_ROCM) $< -o $@

$(LIB_CPU): $(CCOBJECTS_CPU) $(LSTOBJECTS_CPU)
	$(LD_CPU) $(SOFLAGS_CPU) $^ -o $@

$(LIB_CUDA): $(CCOBJECTS_CUDA) $(LSTOBJECTS_CUDA)
	$(LD_CUDA) $(SOFLAGS_CUDA) $^ -o $@

$(LIB_ROCM): $(CCOBJECTS_ROCM) $(LSTOBJECTS_ROCM)
	$(LD_ROCM) $(SOFLAGS_ROCM) $^ -o $@

explicit: $(LIBS)

explicit_cache: CACHEFLAG += $(CACHEFLAG_FLAGS)
explicit_cache: $(LIBS)

explicit_cache_cutvalue: CUTVALUEFLAG = $(CUTVALUEFLAG_FLAGS)
explicit_cache_cutvalue: CACHEFLAG += $(CACHEFLAG_FLAGS)
explicit_cache_cutvalue: $(LIBS)

clean:
	rm -f *.opp
	rm -f *.o
	rm -f *.d
	rm -f *.so

.PHONY: clean explicit explicit_cache explicit_cache_cutvalue format check check-fix

format:
	clang-format --style=file:../.clang-format -i *.cc *.h

# Collect all the include paths from the compiler.
# The .../gcc/x86_64-redhat-linux-gnu/*/include path is excluded since .../gcc/x86_64-redhat-linux-gnu/*/include-fixed should be used instead.
TIDYINCLUDEFLAGS := $(shell g++ -E -x c++ - -v < /dev/null 2>&1 | awk '/#include <...>/,/^End of search/{if (/^ / && !/x86_64-redhat-linux-gnu\/[0-9.]+\/include$$/) print "-I"$$1}' | tr '\n' ' ')
TIDYFLAGS := --language=c++ $(CXXFLAGS_CPU) $(ALPAKAINCLUDE) $(ALPAKASERIAL) $(ROOTCFLAGS) $(PRINTFLAG) $(DUPLICATED) $(CACHEFLAG_FLAGS) $(TIDYINCLUDEFLAGS)

check:
	clang-tidy --config-file=../.clang-tidy *.cc *.h -- $(TIDYFLAGS)

check-fix:
	clang-tidy --config-file=../.clang-tidy --format-style=file:../.clang-format --fix --fix-errors --fix-notes *.cc *.h -- $(TIDYFLAGS)
