#! /bin/bash
function die { echo $1: status $2 ; exit $2; }

echo -e "\n\nTESTING eopTreeWriter (Pion Analysis) ..."
cmsRun ${LOCAL_TEST_DIR}/eopTreeWriter_cfg.py unitTest=True maxEvents=100 || die "Failure running eopTreeWriter" $?

echo -e "\n\nTESTING eopElecTreeWriter (Electron Analysis) ..."
cmsRun ${LOCAL_TEST_DIR}/eopElecTreeWriter_cfg.py maxEvents=100 || die "Failure running eopElecTreeWriter" $?
