#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

cmsRun ${LOCAL_TEST_DIR}/rerunTauRecoOnMiniAOD.py || die 'Failure using rerunTauRecoOnMiniAOD.py' $?
cmsRun ${LOCAL_TEST_DIR}/runDeepTauIDsOnMiniAOD.py || die 'Failure using runDeepTauIDsOnMiniAOD.py' $?
cmsRun ${LOCAL_TEST_DIR}/rerunMVAIsolationOnMiniAOD_Phase2.py || die 'Failure using rerunMVAIsolationOnMiniAOD_Phase2.py' $?
