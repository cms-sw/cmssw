#!/bin/sh
# Script to link AOT products with CUDA support, change hardcoded paths to appropriate architecture

g++ model.o external.o \
    -DTORCH_INDUCTOR_CPP_WRAPPER \
    -DSTANDALONE_TORCH_HEADER \
    -DC10_USING_CUSTOM_GENERATED_MACROS \
    -DCPU_CAPABILITY_AVX512 \
    -D USE_CUDA \
    -shared \
    -fPIC \
    -O3 \
    -DNDEBUG \
    -fno-trapping-math \
    -funsafe-math-optimizations \
    -ffinite-math-only \
    -fno-signed-zeros \
    -fno-math-errno \
    -fexcess-precision=fast \
    -fno-finite-math-only \
    -fno-unsafe-math-optimizations \
    -ffp-contract=off \
    -fno-tree-loop-vectorize \
    -march=native \
    -Wall \
    -std=c++20 \
    -Wno-unused-variable \
    -Wno-unknown-pragmas \
    -fopenmp \
    -L/cvmfs/cms.cern.ch/el8_amd64_gcc12/external/pytorch/2.6.0-7036681b61fcb4a817e8ec934bd569b7/lib \
    -L/cvmfs/cms.cern.ch/el8_amd64_gcc12/external/cuda/12.8.0-15bfa86985d46d842bb5ecc3aca6c676/lib64 \
    -ltorch_cpu \
    -lgomp \
    -lc10_cuda \
    -lcuda \
    -ltorch_cuda \
    -mfma \
    -o model.so
