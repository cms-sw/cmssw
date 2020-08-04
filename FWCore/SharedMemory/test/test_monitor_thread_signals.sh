#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

dir=${PWD}
cd ${LOCALTOP}

#need to find a test program which we do not put on the PATH
${CMSSW_BASE}/test/${SCRAM_ARCH}/testFWCoreSharedMemoryMonitorThread 6 >& testFWCoreSharedMemoryMonitorThread.log && die "did not return signal" 1

retValue=$?
if [ "$retValue" != "134" ]; then
  echo "Wrong return value " $retValue
  exit $retValue
else
  grep -q 'Action run' testFWCoreSharedMemoryMonitorThread.log || die 'Action not run' $?
  grep -q 'Worker: SIGNAL CAUGHT 6' testFWCoreSharedMemoryMonitorThread.log || die 'Signal not reported' $?
fi


${CMSSW_BASE}/test/${SCRAM_ARCH}/testFWCoreSharedMemoryMonitorThread 11 >& testFWCoreSharedMemoryMonitorThread.log && die "did not return signal" 1

retValue=$?
if [ "$retValue" != "139" ]; then
  echo "Wrong return value " $retValue
  exit $retValue
else
  grep -q 'Action run' testFWCoreSharedMemoryMonitorThread.log || die 'Action not run' $?
  grep -q 'Worker: SIGNAL CAUGHT 11' testFWCoreSharedMemoryMonitorThread.log || die 'Signal not reported' $?
fi

