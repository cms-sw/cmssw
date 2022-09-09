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

runSuccess ""
runSuccess "-- --accelerators=cpu"
runSuccess "-- --moduleBackend=serial_sync"

if [ "${TARGET}" == "cpu" ]; then
    runFailure "-- --moduleBackend=cuda_async"

elif [ "${TARGET}" == "cuda" ]; then
    runSuccess "-- --moduleBackend=cuda_async"
fi
