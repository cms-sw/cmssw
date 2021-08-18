#!/bin/sh
set -x
LOCAL_TEST_DIR=${CMSSW_BASE}/src/FWCore/Integration/test
LOCAL_TMP_DIR=${CMSSW_BASE}/tmp/${SCRAM_ARCH}

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

pushd ${LOCAL_TMP_DIR}

cmsRun ${LOCAL_TEST_DIR}/CatchCmsExceptiontest_cfg.py &> CatchCmsException.log && die 'Failed in using CatchCmsException_cfg.py' 1

grep -q WhatsItESProducer CatchCmsException.log || die 'Failed to find Producers name' $?

#echo running cmsRun testSkipEvent_cfg.py
#cmsRun ${LOCAL_TEST_DIR}/testSkipEvent_cfg.py &> testSkipEvent.log || die 'Failed in using testSkipEvent_cfg.py' $?

#echo running cmsRun CatchCmsExceptionFromSource_cfg.py

#cmsRun ${LOCAL_TEST_DIR}/CatchCmsExceptionFromSource_cfg.py &> CatchCmsExceptionFromSource.log && \
#die 'Failed because expected exception was not thrown while running cmsRun CatchCmsExceptionFromSource_cfg.py' 1

#grep -q "Calling Source::beginRun" CatchCmsExceptionFromSource.log || die 'Failed to find string Calling Source::beginRun' $?

# It is intentional that this test throws an exception. The test fails if it does not.
cmsRun ${LOCAL_TEST_DIR}/testMissingDictionaryChecking_cfg.py &> testMissingDictionaryChecking.log && die 'Failed to get exception running testMissingDictionaryChecking_cfg.py' 1
grep -q MissingDictionaryTestF testMissingDictionaryChecking.log || die 'Failed to print out exception message with missing dictionary listed' $?

popd

#grep -w ESProducer CatcheStdException.log
