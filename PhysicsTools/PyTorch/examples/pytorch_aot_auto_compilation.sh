#!/bin/bash
# AOT model compilation script for PyTorch in CMSSW

set -e
function die { echo Failed $1: status $2 ; exit $2 ; }
ARCH=$(scram arch)

# compilation support
if [[ "$1" =~ ^(cpu|cuda)$ ]]; then
  TARGET=$1
else
  die "Argument needs to be 'cpu' or 'cuda'; got '$1'" 1
fi

# model type
if [[ "$2" =~ ^(classification|regression)$ ]]; then
  MODEL=$2
else
  die "Argument needs to be 'classification' or 'regression'; got '$2'" 1
fi

# output artifact name
OUTPUT="tmp"

if [[ "${TARGET}" == "cuda" ]]; then
  # Torch requires Triton for GPU builds, which requires a minimum glibc version of 2.34.
  GLIBC_VERSION=$(ldd --version | head -n1 | grep -oE '[0-9]+\.[0-9]+')
  REQUIRED_GLIBC="2.34"
  version_ge() {
    [ "$(printf '%s\n' "$1" "$2" | sort -V | head -n1)" = "$2" ]
  }
  if ! version_ge "$GLIBC_VERSION" "$REQUIRED_GLIBC"; then
    echo "ERROR: GLIBC $REQUIRED_GLIBC or newer is required. Found $GLIBC_VERSION."
    die "GLIBC version check failed" 1
  else
    echo "GLIBC version $GLIBC_VERSION OK."
  fi
fi

echo "Compiling AOT model for target: ${TARGET}, model type: ${MODEL}"

# export model
python3 python/aot_inductor_export.py --model "${MODEL}" --device "${TARGET}" --output "${OUTPUT}"
echo "OK - Exporting model: ${MODEL} for target: ${TARGET}"

# unzip and prepare model
unzip -q "${OUTPUT}".pt2 -d "${OUTPUT}"
cd "${OUTPUT}"/data/aotinductor/model
./../../../../scripts/pytorch_aot_rename.sh .
echo "OK - Unzipping and preparing model"

# compilation with cmssw
if [ "${TARGET}" == "cpu" ]; then
  echo "Compilation for CPU backend"
  ./../../../../scripts/pytorch_aot_compile.sh
  ./../../../../scripts/pytorch_aot_link.sh
elif [ "${TARGET}" == "cuda" ]; then
  echo "Compilation for CUDA backend"
  ./../../../../scripts/pytorch_aot_compile_cuda.sh
  ./../../../../scripts/pytorch_aot_link_cuda.sh
fi
echo "OK - Compiling model with CMSSW toolchain"

# package model shared library
cd ../../../
zip -qr "${MODEL}_${TARGET}_${ARCH}".pt2 data/*
echo "OK - Packaging model shared library..."

# move to data registry
mv "${MODEL}_${TARGET}_${ARCH}".pt2 ../data/
cd ../
echo "OK - Moving library to data registry"

# cleanup
[ -d "$OUTPUT" ] && rm -r "$OUTPUT"
[ -f "${OUTPUT}".pt2 ] && rm "${OUTPUT}".pt2
echo "OK - Cleanup completed"
echo "AOT model compilation and packaging completed successfully."
