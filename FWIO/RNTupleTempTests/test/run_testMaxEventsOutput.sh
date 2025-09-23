#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

LOCAL_TEST_DIR="${SCRAM_TEST_PATH}"

# Test maxEvents output parameter
F3=${LOCAL_TEST_DIR}/testMaxEventsOutput_cfg.py
echo $F3
(cmsRun $F3 ) || die "Failure running cmsRun $F3" $?
# 6th word on the line containing the string "events"
# output by edmRNTupleTempFileUtil
nEvents=`edmRNTupleTempFileUtil file:testMaxEventsOutput.root | grep events | awk ' {print $6; exit} '`
if [ "$nEvents" -lt 6 ] || [ "$nEvents" -gt 9 ]; then
echo "maxEvents output test failed, nEvents = " $nEvents
exit 1
fi
echo "number of events written = " $nEvents

exit 0
