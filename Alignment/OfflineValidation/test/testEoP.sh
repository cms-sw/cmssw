#! /bin/bash
function die { echo $1: status $2 ; exit $2; }

echo "TESTING DiMuonVertexValidation ..."
cmsRun ${LOCAL_TEST_DIR}/DiMuonVertexValidation_cfg.py maxEvents=10 || die "Failure running DiMuonVertexValidation_cfg.py" $?

echo "TESTING inspectData ..."
cmsRun ${LOCAL_TEST_DIR}/inspectData_cfg.py unitTest=True || die "Failure running inspectData_cfg.py" $?

echo "TESTING eopTreeWriter (Pion Analysis) ..."
cmsRun ${LOCAL_TEST_DIR}/eopTreeWriter_cfg.py unitTest=True maxEvents=10 || die "Failure running eopTreeWriter" $?

echo "TESTING eopElecTreeWriter (Electron Analysis) ..."
cmsRun ${LOCAL_TEST_DIR}/eopElecTreeWriter_cfg.py maxEvents=10 || die "Failure running eopElecTreeWriter" $?
