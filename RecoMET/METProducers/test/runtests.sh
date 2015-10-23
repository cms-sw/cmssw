#!/bin/bash

##____________________________________________________________________________||
function die { echo $1: status $2 ;  exit $2; }

##____________________________________________________________________________||

# temporary: produce fastsim sample on the fly
# can be removed as soon as relval samples are available with the new fastsim rechits
cmsDriver.py TTbar_13TeV_TuneCUETP8M1_cfi  --conditions auto:run2_mc --fast  -n 1 --eventcontent FEVTDEBUGHLT -s GEN,SIM,RECOBEFMIX,DIGI:pdigi_valid,L1,L1Reco,RECO --beamspot NominalCollision2015 --customise SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1 --fileout ttbarForMetTests.root
cmsRun ${LOCAL_TEST_DIR}/recoMET_pfMet_cfg.py || die 'Failure using recoMET_pfMet_cfg.py' $?
cmsRun ${LOCAL_TEST_DIR}/recoMET_caloMet_cfg.py || die 'Failure using recoMET_caloMet_cfg.py' $?
cmsRun ${LOCAL_TEST_DIR}/recoMET_tcMet_cfg.py || die 'Failure using recoMET_tcMet_cfg.py' $?
cmsRun ${LOCAL_TEST_DIR}/recoMET_genMet_cfg.py || die 'Failure using recoMET_genMet_cfg.py' $?
cmsRun ${LOCAL_TEST_DIR}/recoMET_pfChMet_cfg.py || die 'Failure using recoMET_pfChMet_cfg.py' $?
cmsRun ${LOCAL_TEST_DIR}/recoMET_pfClusterMet_cfg.py || die 'Failure using recoMET_pfClusterMet_cfg.py' $?
