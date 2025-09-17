#!/bin/sh
set -x
LOCAL_TEST_DIR=${SCRAM_TEST_PATH}
# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

cmsRun ${LOCAL_TEST_DIR}/CatchCmsExceptiontest_cfg.py &> CatchCmsException.log && die 'Failed in using CatchCmsException_cfg.py' 1

grep -q WhatsItESProducer CatchCmsException.log || die 'Failed to find Producers name' $?

#echo running cmsRun testSkipEvent_cfg.py
#cmsRun ${LOCAL_TEST_DIR}/testSkipEvent_cfg.py &> testSkipEvent.log || die 'Failed in using testSkipEvent_cfg.py' $?
