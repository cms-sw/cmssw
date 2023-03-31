#! /bin/bash

function die { echo $1: status $2 ; exit $2; }

echo "TESTING Primary Vertex Validation: phase-1 setup ..."
cmsRun ${CMSSW_BASE}/src/Alignment/OfflineValidation/test/testPrimaryVertexRelatedValidations_cfg.py isPhase2=False maxEvents=100 || die "Failure running testPrimaryVertexRelatedValidations_cfg.py isPhase2=False" $?

echo "TESTING Primary Vertex Validation: phase-2 setup ..."
cmsRun ${CMSSW_BASE}/src/Alignment/OfflineValidation/test/testPrimaryVertexRelatedValidations_cfg.py isPhase2=True maxEvents=10 || die "Failure running testPrimaryVertexRelatedValidations_cfg.py isPhase2=True" $?
