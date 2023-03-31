#! /bin/bash

function die { echo $1: status $2 ; exit $2; }

echo "TESTING Double Muon Vertex validation ..."
cmsRun ${CMSSW_BASE}/src/Alignment/OfflineValidation/test/DiMuonVertexValidation_cfg.py maxEvents=10 || die "Failure running DiMuonVertexValidation_cfg.py" $?
