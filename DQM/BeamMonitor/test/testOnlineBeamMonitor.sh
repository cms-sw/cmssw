#! /bin/bash

function die { echo $1: status $2 ; exit $2; }

echo "TESTING OnlineBeamMonitor ..."
cmsRun ${LOCAL_TEST_DIR}/Online_BeamMonitor_file.py maxEvents=10 || die "Failure running Online_BeamMonitor_file.py" $?
