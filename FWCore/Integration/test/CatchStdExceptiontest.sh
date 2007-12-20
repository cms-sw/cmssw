#!/bin/sh

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

cmsRun ${LOCAL_TEST_DIR}/CatchStdExceptiontest.cfg &> CatchStdException.log && die 'Failed in using CatchStdException.cfg' $? 

grep -q WhatsItESProducer CatchStdException.log || die 'Failed to find Producers name' $?





#grep -w ESProducer CatcheStdException.log
