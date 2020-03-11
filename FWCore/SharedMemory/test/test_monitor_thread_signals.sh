#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

dir=${PWD}
cd ${LOCALTOP}

#test programs are placed on PATH when tests are run by scram
testpath="testFWCoreSharedMemoryMonitorThread"

$testpath 6 >& testFWCoreSharedMemoryMonitorThread.log && die "did not return signal" 1

retValue=$?
if [ "$retValue" != "134" ]; then
  echo "Wrong return value " $retValue
  exit $retValue
else
  grep -q 'Action run' testFWCoreSharedMemoryMonitorThread.log || die 'Action not run' $?
  grep -q 'Worker: SIGNAL CAUGHT 6' testFWCoreSharedMemoryMonitorThread.log || die 'Signal not reported' $?
fi


$testpath 11 >& testFWCoreSharedMemoryMonitorThread.log && die "did not return signal" 1

retValue=$?
if [ "$retValue" != "139" ]; then
  echo "Wrong return value " $retValue
  exit $retValue
else
  grep -q 'Action run' testFWCoreSharedMemoryMonitorThread.log || die 'Action not run' $?
  grep -q 'Worker: SIGNAL CAUGHT 11' testFWCoreSharedMemoryMonitorThread.log || die 'Signal not reported' $?
fi

