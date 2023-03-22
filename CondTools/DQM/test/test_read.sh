#! /bin/bash

function die { echo $1: status $2 ; exit $2; }

echo "TESTING CondTools/DQM ..."
cmsRun ${SCRAM_TEST_PATH}/DQMXMLFileEventSetupAnalyzer_cfg.py unitTest=True || die "Failure running testCondToolsDQMRead" $?
