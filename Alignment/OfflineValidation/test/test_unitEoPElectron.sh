#! /bin/bash
function die { echo $1: status $2 ; exit $2; }

echo "TESTING eopElecTreeWriter (Electron Analysis) ..."
cmsRun ${LOCAL_TEST_DIR}/eopElecTreeWriter_cfg.py maxEvents=10 || die "Failure running eopElecTreeWriter" $?
