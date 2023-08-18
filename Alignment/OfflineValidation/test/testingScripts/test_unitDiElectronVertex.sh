#! /bin/bash

function die { echo $1: status $2 ; exit $2; }

echo "TESTING Double Electron Vertex validation ..."
cmsRun ${CMSSW_BASE}/src/Alignment/OfflineValidation/test/DiElectronVertexValidation_cfg.py maxEvents=10 || die "Failure running DiElectronVertexValidation_cfg.py" $?
