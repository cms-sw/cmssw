#!/bin/bash

##____________________________________________________________________________||
function die { echo $1: status $2 ;  exit $2; }

##____________________________________________________________________________||
cmsRun ${LOCAL_TEST_DIR}/recoMET_pfMet_cfg.py || die 'Failure using recoMET_pfMet_cfg.py' $?
cmsRun ${LOCAL_TEST_DIR}/recoMET_caloMet_cfg.py || die 'Failure using recoMET_caloMet_cfg.py' $?
cmsRun ${LOCAL_TEST_DIR}/recoMET_tcMet_cfg.py || die 'Failure using recoMET_tcMet_cfg.py' $?
cmsRun ${LOCAL_TEST_DIR}/recoMET_genMet_cfg.py || die 'Failure using recoMET_genMet_cfg.py' $?
cmsRun ${LOCAL_TEST_DIR}/recoMET_pfChargedMET_cfg.py || die 'Failure using recoMET_pfChargedMET_cfg.py' $?
cmsRun ${LOCAL_TEST_DIR}/recoMET_htMet_cfg.py || die 'Failure using recoMET_htMet_cfg.py' $?
cmsRun ${LOCAL_TEST_DIR}/recoMET_pfClusterMet_cfg.py || die 'Failure using recoMET_pfClusterMet_cfg.py' $?
