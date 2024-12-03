#
# stuff to make
#

CCSOURCES=$(wildcard ../../src/*.cc)
ALPAKACCSOURCES=$(wildcard ../../src/alpaka/*.dev.cc)
CCOBJECTS_CPU=$(patsubst ../../src/alpaka/%.dev.cc, %_cpu.o, $(ALPAKACCSOURCES)) $(patsubst ../../src/%.cc, %_cpu.o, $(CCSOURCES))
CCOBJECTS_CUDA=$(patsubst ../../src/alpaka/%.dev.cc, %_cuda.o, $(ALPAKACCSOURCES)) $(patsubst ../../src/%.cc, %_cuda.o, $(CCSOURCES))
CCOBJECTS_ROCM=$(patsubst ../../src/alpaka/%.dev.cc, %_rocm.o, $(ALPAKACCSOURCES)) $(patsubst ../../src/%.cc, %_rocm.o, $(CCSOURCES))

LSTSOURCES=../../src/alpaka/LST.cc
LSTOBJECTS_CPU=$(patsubst ../../src/alpaka/%.cc, %_cpu.o, $(LSTSOURCES))
LSTOBJECTS_CUDA=$(patsubst ../../src/alpaka/%.cc, %_cuda.o, $(LSTSOURCES))
LSTOBJECTS_ROCM=$(patsubst ../../src/alpaka/%.cc, %_rocm.o, $(LSTSOURCES))

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
CXXFLAGS_CPU         = -march=native -mtune=native -Ofast -fno-reciprocal-math -fopenmp-simd -g -Wall -Woverloaded-virtual -fPIC -fopenmp -I..
CXXFLAGS_CUDA        = -O3 -g --compiler-options -Wall --compiler-options -Woverloaded-virtual --compiler-options -fPIC --compiler-options -fopenmp -dc -lineinfo --ptxas-options=-v --cudart shared $(GENCODE_CUDA) --use_fast_math --default-stream per-thread -I..
CXXFLAGS_ROCM        = -O3 -g -Wall -Woverloaded-virtual -fPIC -I${ROCM_ROOT}/include -I..
CMSSWINCLUDE        := -I${TRACKLOOPERDIR}/../../../ -I${CMSSW_BASE}/src
ifdef CMSSW_RELEASE_BASE
CMSSWINCLUDE        := ${CMSSWINCLUDE} -I${CMSSW_RELEASE_BASE}/src
endif
ALPAKAINCLUDE        = -I${ALPAKA_ROOT}/include -I/${BOOST_ROOT}/include -std=c++20 ${CMSSWINCLUDE}
ALPAKASERIAL         = -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED -DALPAKA_DISABLE_VENDOR_RNG -DALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128
ALPAKACUDA           = -DALPAKA_ACC_GPU_CUDA_ENABLED -DALPAKA_ACC_GPU_CUDA_ONLY -DALPAKA_DISABLE_VENDOR_RNG -DALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128 --expt-relaxed-constexpr
ALPAKAROCM           = -DALPAKA_ACC_GPU_HIP_ENABLED -DALPAKA_ACC_GPU_HIP_ONLY -DALPAKA_DISABLE_VENDOR_RNG -DALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128 -DGNU_GCC -D_GNU_SOURCE --target=x86_64-redhat-linux-gnu --gcc-toolchain=$(patsubst %/bin/gcc,%,$(shell command -v gcc))
ROOTINCLUDE          = -I$(ROOT_ROOT)/include
ROOTCFLAGS           = -pthread -m64 $(ROOTINCLUDE)
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
T5CUTFLAGS           = $(T5DNNFLAG) $(T5RZCHI2FLAG)

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
	$(COMPILE_CMD_CPU) $(CXXFLAGS_CPU) $(ROOTINCLUDE) $(CUTVALUEFLAG) $(LSTWARNINGSFLAG) $(CMSSW_WERRORS_CPU) $(T5CUTFLAGS) $(ALPAKAINCLUDE) $(ALPAKABACKEND_CPU) $< -o $@

%_cuda.o: ../../src/alpaka/%.dev.cc
	$(COMPILE_CMD_CUDA) $(CXXFLAGS_CUDA) $(ROOTINCLUDE) $(CUTVALUEFLAG) $(LSTWARNINGSFLAG) $(CMSSW_WERRORS_CUDA) $(T5CUTFLAGS) $(ALPAKAINCLUDE) $(ALPAKABACKEND_CUDA) $< -o $@

%_rocm.o: ../../src/alpaka/%.dev.cc
	$(COMPILE_CMD_ROCM) $(CXXFLAGS_ROCM) $(ROOTINCLUDE) $(CUTVALUEFLAG) $(LSTWARNINGSFLAG) $(CMSSW_WERRORS_ROCM) $(T5CUTFLAGS) $(ALPAKAINCLUDE) $(ALPAKABACKEND_ROCM) $< -o $@

%_cpu.o: ../../src/alpaka/%.cc
	$(COMPILE_CMD_CPU) $(CXXFLAGS_CPU) $(ROOTINCLUDE) $(CUTVALUEFLAG) $(LSTWARNINGSFLAG) $(CMSSW_WERRORS_CPU) $(T5CUTFLAGS) $(ALPAKAINCLUDE) $(ALPAKABACKEND_CPU) $< -o $@

%_cuda.o: ../../src/alpaka/%.cc
	$(COMPILE_CMD_CUDA) $(CXXFLAGS_CUDA) $(ROOTINCLUDE) $(CUTVALUEFLAG) $(LSTWARNINGSFLAG) $(CMSSW_WERRORS_CUDA) $(T5CUTFLAGS) $(ALPAKAINCLUDE) $(ALPAKABACKEND_CUDA) $< -o $@

%_rocm.o: ../../src/alpaka/%.cc
	$(COMPILE_CMD_ROCM) $(CXXFLAGS_ROCM) $(ROOTINCLUDE) $(CUTVALUEFLAG) $(LSTWARNINGSFLAG) $(CMSSW_WERRORS_ROCM) $(T5CUTFLAGS) $(ALPAKAINCLUDE) $(ALPAKABACKEND_ROCM) $< -o $@

%_cpu.o: ../../src/%.cc
	$(COMPILE_CMD_CPU) $(CXXFLAGS_CPU) $(ROOTINCLUDE) $(CUTVALUEFLAG) $(LSTWARNINGSFLAG) $(CMSSW_WERRORS_CPU) $(T5CUTFLAGS) $(ALPAKAINCLUDE) $(ALPAKABACKEND_CPU) $< -o $@

%_cuda.o: ../../src/%.cc
	$(COMPILE_CMD_CUDA) $(CXXFLAGS_CUDA) $(ROOTINCLUDE) $(CUTVALUEFLAG) $(LSTWARNINGSFLAG) $(CMSSW_WERRORS_CUDA) $(T5CUTFLAGS) $(ALPAKAINCLUDE) $(ALPAKABACKEND_CUDA) $< -o $@

%_rocm.o: ../../src/%.cc
	$(COMPILE_CMD_ROCM) $(CXXFLAGS_ROCM) $(ROOTINCLUDE) $(CUTVALUEFLAG) $(LSTWARNINGSFLAG) $(CMSSW_WERRORS_ROCM) $(T5CUTFLAGS) $(ALPAKAINCLUDE) $(ALPAKABACKEND_ROCM) $< -o $@

$(LIB_CPU): $(CCOBJECTS_CPU) $(LSTOBJECTS_CPU)
	$(LD_CPU) $(SOFLAGS_CPU) $^ -o $@

$(LIB_CUDA): $(CCOBJECTS_CUDA) $(LSTOBJECTS_CUDA)
	$(LD_CUDA) $(SOFLAGS_CUDA) $^ -o $@

$(LIB_ROCM): $(CCOBJECTS_ROCM) $(LSTOBJECTS_ROCM)
	$(LD_ROCM) $(SOFLAGS_ROCM) $^ -o $@

explicit: $(LIBS)

explicit_cutvalue: CUTVALUEFLAG = $(CUTVALUEFLAG_FLAGS)
explicit_cutvalue: $(LIBS)

clean:
	rm -f *.opp
	rm -f *.o
	rm -f *.d
	rm -f *.so

.PHONY: clean explicit explicit_cutvalue format check check-fix

format:
	clang-format --style=file:../.clang-format -i *.cc *.h

# Collect all the include paths from the compiler.
# The .../gcc/x86_64-redhat-linux-gnu/*/include path is excluded since .../gcc/x86_64-redhat-linux-gnu/*/include-fixed should be used instead.
TIDYINCLUDEFLAGS := $(shell g++ -E -x c++ - -v < /dev/null 2>&1 | awk '/#include <...>/,/^End of search/{if (/^ / && !/x86_64-redhat-linux-gnu\/[0-9.]+\/include$$/) print "-I"$$1}' | tr '\n' ' ')
TIDYFLAGS := --language=c++ $(CXXFLAGS_CPU) $(ALPAKAINCLUDE) $(ALPAKASERIAL) $(ROOTCFLAGS) $(DUPLICATED) $(TIDYINCLUDEFLAGS)

check:
	clang-tidy --config-file=../.clang-tidy *.cc *.h -- $(TIDYFLAGS)

check-fix:
	clang-tidy --config-file=../.clang-tidy --format-style=file:../.clang-format --fix --fix-errors --fix-notes *.cc *.h -- $(TIDYFLAGS)
