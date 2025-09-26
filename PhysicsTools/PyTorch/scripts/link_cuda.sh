#!/bin/sh
# Script to link AOT products with CUDA support, change hardcoded paths to appropriate architecture

TORCH=$(scram tool tag torch-interface TORCH_INTERFACE_BASE)
CUDA=$(scram tool tag cuda CUDA_BASE)

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
    -L${TORCH}/lib \
    -L${CUDA}/lib64 \
    -ltorch_cpu \
    -lgomp \
    -lc10_cuda \
    -lcuda \
    -ltorch_cuda \
    -mfma \
    -o model.so
