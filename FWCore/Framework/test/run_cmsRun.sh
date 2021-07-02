#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

(cmsRun --help ) || die 'Failure running cmsRun --help' $?

# This test is supposed to throw an exception.
# We had a bug where EventProcessor went into an
# infinite wait under the circumstances in this test
# and after an exception. The conditions were multiple
# concurrent lumis in flight with an exception on an
# event in a lumi before the last lumi.
# This test passes as long as it does not go into
# an infinite wait.
F2=${LOCAL_TEST_DIR}/testConcurrentLumiExceptions_cfg.py
echo $F2 "This test intentionally throws an exception"
(cmsRun $F2 ) && die "No exception using $F2" 1

# Test maxEvents output parameter
F3=${LOCAL_TEST_DIR}/testMaxEventsOutput_cfg.py
echo $F3
(cmsRun $F3 ) || die "Failure running cmsRun $F3" $?
# 6th word on the line containing the string "events"
# output by edmFileUtil
nEvents=`edmFileUtil file:testMaxEventsOutput.root | grep events | awk ' {print $6; exit} '`
if [ "$nEvents" -lt 6 ] || [ "$nEvents" -gt 9 ]; then
echo "maxEvents output test failed, nEvents = " $nEvents
exit 1
fi
echo "number of events written = " $nEvents

exit 0
