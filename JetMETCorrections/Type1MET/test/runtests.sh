#!/bin/bash

##____________________________________________________________________________||
function die { echo $1: status $2 ;  exit $2; }

##____________________________________________________________________________||
cmsRun ${SCRAM_TEST_PATH}/corrMET_pfMet_cfg.py || die 'Failure using corrMET_pfMet_cfg.py' $?
cmsRun ${SCRAM_TEST_PATH}/corrMET_caloMet_cfg.py || die 'Failure using corrMET_caloMet_cfg.py' $?
