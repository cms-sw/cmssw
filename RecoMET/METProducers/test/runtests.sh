#!/bin/bash -ex

##____________________________________________________________________________||
function die { echo $1: status $2 ;  exit $2; }

##____________________________________________________________________________||

# temporary: produce fastsim sample on the fly
# can be removed as soon as relval samples are available with the new fastsim rechits
cmsDriver.py TTbar_13TeV_TuneCUETP8M1_cfi  --conditions auto:run2_mc --fast  -n 1 --eventcontent FEVTDEBUGHLT -s GEN,SIM,RECOBEFMIX,DIGI:pdigi_valid,L1,L1Reco,RECO --beamspot NominalCollision2015 --era Run2_2016 --fileout ttbarForMetTests.root || die 'Failure running cmsDriver' $?
cmsRun ${SCRAM_TEST_PATH}/recoMET_pfMet_cfg.py || die 'Failure using recoMET_pfMet_cfg.py' $?
cmsRun ${SCRAM_TEST_PATH}/recoMET_caloMet_cfg.py || die 'Failure using recoMET_caloMet_cfg.py' $?
cmsRun ${SCRAM_TEST_PATH}/recoMET_tcMet_cfg.py || die 'Failure using recoMET_tcMet_cfg.py' $?
cmsRun ${SCRAM_TEST_PATH}/recoMET_genMet_cfg.py || die 'Failure using recoMET_genMet_cfg.py' $?
cmsRun ${SCRAM_TEST_PATH}/recoMET_pfChMet_cfg.py || die 'Failure using recoMET_pfChMet_cfg.py' $?
cmsRun ${SCRAM_TEST_PATH}/recoMET_pfClusterMet_cfg.py || die 'Failure using recoMET_pfClusterMet_cfg.py' $?
