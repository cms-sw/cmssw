#! /bin/bash
function die { echo $1: status $2 ; exit $2; }

echo "TESTING eopTreeWriter (Pion Analysis) ..."
cmsRun ${LOCAL_TEST_DIR}/eopTreeWriter_cfg.py unitTest=True maxEvents=10 || die "Failure running eopTreeWriter" $?
