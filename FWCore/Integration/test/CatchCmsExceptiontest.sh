#!/bin/sh
set -x
LOCAL_TEST_DIR=${SCRAM_TEST_PATH}
# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

function test_failure { 
    if [ "$2" != "65" ]
     then
       echo $1: status $2; exit $2;
    fi
}


cmsRun ${LOCAL_TEST_DIR}/CatchCmsExceptiontest_cfg.py &> CatchCmsException.log && die 'Failed in using CatchCmsException_cfg.py' 1

grep -q WhatsItESProducer CatchCmsException.log || die 'Failed to find Producers name' $?

#This is the case where the exception is not thrown, so the exit code should be 0
cmsRun ${LOCAL_TEST_DIR}/CatchCmsExceptionFromSource_cfg.py --whenToThrow=0 || die 'Failed in using CatchCmsExceptionFromSource_cfg.py' $?

for whenToThrow in $(seq 1 11); do
  cmsRun ${LOCAL_TEST_DIR}/CatchCmsExceptionFromSource_cfg.py --whenToThrow=${whenToThrow}; test_failure "Failed in using CatchCmsExceptionFromSource_cfg.py --whenToThrow=${whenToThrow}" $?
done
