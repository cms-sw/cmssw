#!/bin/bash

function die { echo Failed $1: status $2 ; exit $2 ; }

TEST_DIR=${LOCALTOP}/src/HeterogeneousCore/AlpakaTest/test

if [ "$#" != "1" ]; then
    die "Need exactly 1 argument ('cpu', 'cuda'), got $#" 1
fi
if [ "$1" = "cuda" ]; then
    TARGET=cuda
elif [ "$1" = "cpu" ]; then
    # In non-_GPU_ IBs, if CUDA is enabled, run the GPU-targeted tests
    cudaIsEnabled
    CUDA_ENABLED=$?
    if [ "${CUDA_ENABLED}" = "0" ]; then
        TARGET=cuda
    else
        TARGET=cpu
    fi
else
    die "Argument needs to be 'cpu' or 'cuda', got $1" 1
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
    runSuccess "--expectBackend=cuda_async"
    runSuccess "--accelerators=gpu-nvidia --expectBackend=cuda_async"
    runSuccess "--processAcceleratorBackend=cuda_async --expectBackend=cuda_async"
    runSuccess "--moduleBackend=cuda_async --expectBackend=cuda_async"

    runSuccess "--processAcceleratorBackend=cuda_async --moduleBackend=serial_sync --expectBackend=serial_sync"
    runSuccess "--processAcceleratorBackend=serial_sync --moduleBackend=cuda_async --expectBackend=cuda_async"

    runFailure "--accelerators=gpu-nvidia --processAcceleratorBackend=serial_sync --expectBackend=serial_sync"
    runFailure "--accelerators=gpu-nvidia --moduleBackend=serial_sync --expectBackend=serial_sync"
    runFailure "--accelerators=gpu-nvidia --processAcceleratorBackend=cuda_async --moduleBackend=serial_sync --expectBackend=serial_sync"
    runFailure "--accelerators=gpu-nvidia --processAcceleratorBackend=serial_sync --moduleBackend=cuda_async --expectBackend=cuda_async"
    runFailure "--accelerators=cpu --processAcceleratorBackend=cuda_async --expectBackend=cuda_async"
    runFailure "--accelerators=cpu --moduleBackend=cuda_async --expectBackend=cuda_async"
    runFailure "--accelerators=cpu --processAcceleratorBackend=serial_sync --moduleBackend=cuda_async --expectBackend=cuda_async"
    runFailure "--accelerators=cpu --processAcceleratorBackend=cuda_async --moduleBackend=serial_sync --expectBackend=serial_sync"

    runSuccessHostAndDevice "--expectBackend=cuda_async"

fi
