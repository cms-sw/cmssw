#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

F1=${CMSSW_BASE}/src/PhysicsTools/Scouting/test/test_Run3ScoutingElectronBestTrack_cfg.py
(cmsRun $F1 ) || die "Failure runnning $F1" $?

# If the above test passes, test the content of the output ROOT Failure
echo "Testing the content of the output ROOT file"
edmDumpEventContent output_file.root > output_evtctnt.txt
diff output_evtctnt.txt ${CMSSW_BASE}/src/PhysicsTools/Scouting/test/ref_Run3ScoutingElectronBestTrack_outputevtctnt.txt || die "Failure comparing edmDumpEventContent" $?
