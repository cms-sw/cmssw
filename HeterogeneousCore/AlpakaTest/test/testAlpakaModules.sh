#!/bin/bash

function die { echo Failed $1: status $2 ; exit $2 ; }

TEST_DIR=${LOCALTOP}/src/HeterogeneousCore/AlpakaTest/test

if [ "$#" != "1" ]; then
    die "Need exactly 1 argument ('cpu', 'cuda', or 'rocm'), got $#" 1
fi
if [[ "$1" =~ ^(cpu|cuda|rocm)$ ]]; then
    TARGET=$1
else
    die "Argument needs to be 'cpu', 'cuda', or 'rocm'; got '$1'" 1
fi

# Some of the CPU-only tests fail if run on machine with GPU
if [ "$TARGET" == "cpu" ]; then
    cudaIsEnabled
    if [ "$?" == "0" ]; then
        echo "Test target is 'cpu', but NVIDIA GPU is detected. Ignoring the CPU tests."
        exit 0
    fi
    rocmIsEnabled
    if [ "$?" == "0" ]; then
        echo "Test target is 'cpu', but AMD GPU is detected. Ignoring the CPU tests."
        exit 0
    fi
fi

function runSuccess {
    echo "cmsRun testAlpakaModules_cfg.py $1"
    cmsRun ${TEST_DIR}/testAlpakaModules_cfg.py $1 || die "cmsRun testAlpakaModules_cfg.py $1" $?
    echo
}
function runSuccessHostAndDevice {
    echo "cmsRun testAlpakaModulesHostAndDevice_cfg.py $1"
    cmsRun ${TEST_DIR}/testAlpakaModulesHostAndDevice_cfg.py $1 || die "cmsRun testAlpakaModulesHostAndDevice_cfg.py $1" $?
    echo
}
function runFailure {
    echo "cmsRun testAlpakaModules_cfg.py $1 (job itself is expected to fail)"
    cmsRun -j testAlpakaModules_jobreport.xml ${TEST_DIR}/testAlpakaModules_cfg.py $1 && die "cmsRun testAlpakaModules_cfg.py $1 did not fail" 1
    EXIT_CODE=$(edmFjrDump --exitCode testAlpakaModules_jobreport.xml)
    if [ "x${EXIT_CODE}" != "x8035" ]; then
        echo "Alpaka module test for unavailable accelerator reported exit code ${EXIT_CODE} which is different from the expected 8035"
        exit 1
    fi
    echo
}

function runForGPU {
    ACCELERATOR=$1
    BACKEND=$2

    runSuccess "--expectBackend=$BACKEND"
    runSuccess "--accelerators=$ACCELERATOR --expectBackend=$BACKEND"
    runSuccess "--processAcceleratorBackend=$BACKEND --expectBackend=$BACKEND"
    runSuccess "--moduleBackend=$BACKEND --expectBackend=$BACKEND"

    runSuccess "--processAcceleratorBackend=$BACKEND --moduleBackend=serial_sync --expectBackend=serial_sync"
    runSuccess "--processAcceleratorBackend=serial_sync --moduleBackend=$BACKEND --expectBackend=$BACKEND"

    runFailure "--accelerators=$ACCELERATOR --processAcceleratorBackend=serial_sync --expectBackend=serial_sync"
    runFailure "--accelerators=$ACCELERATOR --moduleBackend=serial_sync --expectBackend=serial_sync"
    runFailure "--accelerators=$ACCELERATOR --processAcceleratorBackend=$BACKEND --moduleBackend=serial_sync --expectBackend=serial_sync"
    runFailure "--accelerators=$ACCELERATOR --processAcceleratorBackend=serial_sync --moduleBackend=$BACKEND --expectBackend=$BACKEND"
    runFailure "--accelerators=cpu --processAcceleratorBackend=$BACKEND --expectBackend=$BACKEND"
    runFailure "--accelerators=cpu --moduleBackend=$BACKEND --expectBackend=$BACKEND"
    runFailure "--accelerators=cpu --processAcceleratorBackend=serial_sync --moduleBackend=$BACKEND --expectBackend=$BACKEND"
    runFailure "--accelerators=cpu --processAcceleratorBackend=$BACKEND --moduleBackend=serial_sync --expectBackend=serial_sync"

    runSuccessHostAndDevice "--expectBackend=$BACKEND"
}

runSuccess "--accelerators=cpu --expectBackend=serial_sync"
runSuccess "--processAcceleratorBackend=serial_sync --expectBackend=serial_sync"
runSuccess "--moduleBackend=serial_sync --expectBackend=serial_sync"

if [ "${TARGET}" == "cpu" ]; then
    runSuccess "--expectBackend=serial_sync"

    runFailure "--accelerators=gpu-nvidia --expectBackend=cuda_async"
    runFailure "--processAcceleratorBackend=cuda_async --expectBackend=cuda_async"
    runFailure "--moduleBackend=cuda_async --expectBackend=cuda_async"

    runFailure "--processAcceleratorBackend=cuda_async --moduleBackend=serial_sync --expectBackend=serial_sync"
    runFailure "--processAcceleratorBackend=serial_sync --moduleBackend=cuda_async --expectBackend=cuda_async"

    runSuccessHostAndDevice "--expectBackend=serial_sync"

elif [ "${TARGET}" == "cuda" ]; then
    runForGPU "gpu-nvidia" "cuda_async"

elif [ "${TARGET}" == "rocm" ]; then
    runForGPU "gpu-amd" "rocm_async"

fi
