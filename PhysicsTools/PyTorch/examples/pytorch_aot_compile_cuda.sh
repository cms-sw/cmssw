#!/bin/sh
# Script to compile AOT products with CUDA support, change hardcoded paths to appropriate architecture

PYTHON3=$(scram tool tag python3 PYTHON3_BASE)
TORCH=$(scram tool tag torch-interface TORCH_INTERFACE_BASE)
CUDA=$(scram tool tag cuda CUDA_BASE)

g++ model.cpp \
    -D TORCH_INDUCTOR_CPP_WRAPPER \
    -D STANDALONE_TORCH_HEADER \
    -D C10_USING_CUSTOM_GENERATED_MACROS \
    -D CPU_CAPABILITY_AVX512 \
    -D USE_CUDA \
    -I${PYTHON3}/include/python3.9 \
    -I${TORCH}/include/ \
    -I${TORCH}/include/THC \
    -I${TORCH}/include/torch/csrc/api/include \
    -I${TORCH}/include/TH \
    -I${CUDA}/include \
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
    -c \
    -o model.o
