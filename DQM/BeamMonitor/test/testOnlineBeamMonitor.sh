#! /bin/bash

function die { echo $1: status $2 ; exit $2; }

echo "TESTING OnlineBeamMonitor ..."
cmsRun ${SCRAM_TEST_PATH}/Online_BeamMonitor_file.py maxEvents=10 || die "Failure running Online_BeamMonitor_file.py" $?
