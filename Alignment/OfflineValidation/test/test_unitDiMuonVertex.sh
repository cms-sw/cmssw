#! /bin/bash

function die { echo $1: status $2 ; exit $2; }

echo "TESTING Double Muon Vertex validation ..."
cmsRun ${LOCAL_TEST_DIR}/DiMuonVertexValidation_cfg.py maxEvents=10 || die "Failure running DiMuonVertexValidation_cfg.py" $?
