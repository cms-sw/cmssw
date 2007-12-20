#!/bin/sh

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

cmsRun ${LOCAL_TEST_DIR}/CatchCmsExceptiontest.cfg &> CatchCmsException.log && die 'Failed in using CatchCmsException.cfg' $? 

grep -q WhatsItESProducer CatchCmsException.log || die 'Failed to find Producers name' $?



#grep -w ESProducer CatcheStdException.log
