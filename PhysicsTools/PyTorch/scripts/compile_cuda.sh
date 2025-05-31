#!/bin/sh
# Script to compile AOT products with CUDA support, change hardcoded paths to appropriate architecture

g++ model.cpp \
    -D TORCH_INDUCTOR_CPP_WRAPPER \
    -D STANDALONE_TORCH_HEADER \
    -D C10_USING_CUSTOM_GENERATED_MACROS \
    -D CPU_CAPABILITY_AVX512 \
    -D USE_CUDA \
    -I/cvmfs/cms.cern.ch/el8_amd64_gcc12/external/python3/3.9.14-c10287ae9cadff55334e60003302c349/include/python3.9 \
    -I/cvmfs/cms.cern.ch/el8_amd64_gcc12/external/pytorch/2.6.0-7036681b61fcb4a817e8ec934bd569b7/include/ \
    -I/cvmfs/cms.cern.ch/el8_amd64_gcc12/external/pytorch/2.6.0-7036681b61fcb4a817e8ec934bd569b7/include/THC \
    -I/cvmfs/cms.cern.ch/el8_amd64_gcc12/external/pytorch/2.6.0-7036681b61fcb4a817e8ec934bd569b7/include/torch/csrc/api/include \
    -I/cvmfs/cms.cern.ch/el8_amd64_gcc12/external/pytorch/2.6.0-7036681b61fcb4a817e8ec934bd569b7/include/TH \
    -I/cvmfs/cms.cern.ch/el8_amd64_gcc12/external/cuda/12.8.0-15bfa86985d46d842bb5ecc3aca6c676/include \
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
