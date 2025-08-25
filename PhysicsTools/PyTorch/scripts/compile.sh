#!/bin/sh
# Script to compile AOT products, change hardcoded paths to appropriate architecture

PYTHON3=$(scram tool tag python3 PYTHON3_BASE)
TORCH=$(scram tool tag torch-interface TORCH_INTERFACE_BASE)

g++ model.cpp \
  -DTORCH_INDUCTOR_CPP_WRAPPER \
  -DSTANDALONE_TORCH_HEADER \
  -DC10_USING_CUSTOM_GENERATED_MACROS \
  -DCPU_CAPABILITY_AVX512 \
  -I${PYTHON3}/include/python3.9 \
  -I${TORCH}/include/ \
  -I${TORCH}/include/THC \
  -I${TORCH}/include/torch/csrc/api/include \
  -I${TORCH}/include/TH \
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
