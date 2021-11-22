#!/bin/sh
set -x
LOCAL_TEST_DIR=${CMSSW_BASE}/src/FWCore/Integration/test
LOCAL_TMP_DIR=${CMSSW_BASE}/tmp/${SCRAM_ARCH}

# Pass in name and status
function die { echo $1: status $2 ; echo === Test log === ; cat ${3:-/dev/null} ; echo === End test log === ;  exit $2; }

pushd ${LOCAL_TMP_DIR}

cmsRun ${LOCAL_TEST_DIR}/CatchStdExceptiontest_cfg.py &> CatchStdException.log && die 'Failed in using CatchStdException_cfg.py' 1 CatchStdException.log

grep -q WhatsItESProducer CatchStdException.log || die 'Failed to find Producers name' $? CatchStdException.log
#grep -w ESProducer CatcheStdException.log

popd

