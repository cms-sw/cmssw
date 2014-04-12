#!/bin/bash

##____________________________________________________________________________||
function die { echo $1: status $2 ;  exit $2; }

##____________________________________________________________________________||
cmsRun ${LOCAL_TEST_DIR}/corrMET_pfMet_cfg.py || die 'Failure using corrMET_pfMet_cfg.py' $?
cmsRun ${LOCAL_TEST_DIR}/corrMET_caloMet_cfg.py || die 'Failure using corrMET_caloMet_cfg.py' $?
