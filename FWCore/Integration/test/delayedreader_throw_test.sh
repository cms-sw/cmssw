#!/bin/bash
set -x
LOCAL_TEST_DIR=${CMSSW_BASE}/src/FWCore/Integration/test
LOCAL_TMP_DIR=${CMSSW_BASE}/tmp/${SCRAM_ARCH}

function die { echo Failure $1: status $2 ; exit $2 ; }

pushd ${LOCAL_TMP_DIR}

cmsRun ${LOCAL_TEST_DIR}/delayedreader_throw_cfg.py && die "cmsRun ${test}delayedreader_throw_cfg.py did not fail" 1

popd
exit 0