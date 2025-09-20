#!/bin/bash
 
function die { echo Failed $1: status $2 ; exit $2 ; }

SCRIPT="${LOCALTOP}/src/PhysicsTools/PyTorch/test/testPipeline.py"

if [ "$#" != "1" ]; then
  die "Need exactly 1 argument: ('cpu', 'cuda', or 'rocm') got $#" 1
fi
if [[ "$1" =~ ^(cpu|cuda|rocm)$ ]]; then
  TARGET=$1
else
  die "Argument needs to be 'cpu', 'cuda', or 'rocm'; got '$1'" 1
fi

if [ "${TARGET}" == "cpu" ]; then
  echo "Running CPU-only test"
  cmsRun "${SCRIPT}" backend=serial_sync
elif [ "${TARGET}" == "cuda" ]; then
  echo "Running CUDA test"
  cmsRun "${SCRIPT}" backend=cuda_async
elif [ "${TARGET}" == "rocm" ]; then
  echo "Running ROCm test"
  cmsRun "${SCRIPT}" backend=rocm_async
fi
