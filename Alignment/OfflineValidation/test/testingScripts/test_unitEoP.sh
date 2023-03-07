#! /bin/bash
function die { echo $1: status $2 ; exit $2; }

echo -e "\n\nTESTING eopTreeWriter (Pion Analysis) ..."
cmsRun ${CMSSW_BASE}/src/Alignment/OfflineValidation/test/eopTreeWriter_cfg.py unitTest=True maxEvents=100 || die "Failure running eopTreeWriter" $?

echo -e "\n\nTESTING eopElecTreeWriter (Electron Analysis) ..."
cmsRun ${CMSSW_BASE}/src/Alignment/OfflineValidation/test/eopElecTreeWriter_cfg.py maxEvents=100 || die "Failure running eopElecTreeWriter" $?
