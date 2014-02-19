#!/bin/bash

##____________________________________________________________________________||
function die { echo $1: status $2 ;  exit $2; }

##____________________________________________________________________________||
cmsRun ${LOCAL_TEST_DIR}/recoMET_pfMet_cfg.py || die 'Failure using recoMET_pfMet_cfg.py' $?
