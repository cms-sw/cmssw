#! /bin/bash
function die { echo $1: status $2 ; exit $2; }

echo "TESTING G4e refitter ..."
cmsRun ${LOCAL_TEST_DIR}/testG4Refitter_cfg.py maxEvents=50  || die "Failure running testG4Refitter_cfg.py" $?

echo "TESTING DiMuonVertexValidation ..."
cmsRun ${LOCAL_TEST_DIR}/DiMuonVertexValidation_cfg.py maxEvents=10 || die "Failure running DiMuonVertexValidation_cfg.py" $?

echo "TESTING inspectData ..."
cmsRun ${LOCAL_TEST_DIR}/inspectData_cfg.py unitTest=True || die "Failure running inspectData_cfg.py" $?
