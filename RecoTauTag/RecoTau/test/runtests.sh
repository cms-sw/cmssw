#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

cmsRun ${SCRAM_TEST_PATH}/rerunTauRecoOnMiniAOD.py || die 'Failure using rerunTauRecoOnMiniAOD.py' $?
cmsRun ${SCRAM_TEST_PATH}/runDeepTauIDsOnMiniAOD.py || die 'Failure using runDeepTauIDsOnMiniAOD.py' $?
cmsRun ${SCRAM_TEST_PATH}/rerunMVAIsolationOnMiniAOD_Phase2.py || die 'Failure using rerunMVAIsolationOnMiniAOD_Phase2.py' $?
