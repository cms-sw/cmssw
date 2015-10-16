#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

cmsRun ${LOCAL_TEST_DIR}/patTuple_standard_cfg.py || die 'Failure using patTuple_standard_cfg.py' $?

# FIXME: event content broken in only available data RelVal
# cmsRun ${LOCAL_TEST_DIR}/patTuple_data_cfg.py || die 'Failure using patTuple_data_cfg.py' $? # Recent input RelVals currently not reachable (from CERN)

cmsRun ${LOCAL_TEST_DIR}/patTuple_PF2PAT_cfg.py || die 'Failure using patTuple_PF2PAT_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/patTuple_PATandPF2PAT_cfg.py || die 'Failure using patTuple_PATandPF2PAT_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/patTuple_addDecayInFlight_cfg.py || die 'Failure using patTuple_addDecayInFlight_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/patTuple_addBTagging_cfg.py || die 'Failure using patTuple_addBTagging_cfg.py' $?

#broken now
#cmsRun ${LOCAL_TEST_DIR}/patTuple_addJets_cfg.py || die 'Failure using patTuple_addJets_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/patTuple_addTracks_cfg.py || die 'Failure using patTuple_addTracks_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/patTuple_addTriggerInfo_cfg.py || die 'Failure using patTuple_addTriggerInfo_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/patTuple_addVertexInfo_cfg.py || die 'Failure using patTuple_addVertexInfo_cfg.py' $?

# temporary: produce fastsim sample on the fly
# can be removed as soon as relval samples are available with the new fastsim rechits
cmsDriver.py TTbar_13TeV_TuneCUETP8M1_cfi  --conditions auto:run2_mc --fast  -n 1 --eventcontent FEVTDEBUGHLT -s GEN,SIM,RECOBEFMIX,DIGI:pdigi_valid,L1,L1Reco,RECO --beamspot NominalCollision2015 --customise SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1 --fileout ttbarForFastSimTest.root

cmsRun ${LOCAL_TEST_DIR}/patTuple_fastsim_cfg.py || die 'Failure using patTuple_fastsim_cfg.py' $?

# cmsRun ${LOCAL_TEST_DIR}/patTuple_topSelection_cfg.py || die 'Failure using patTuple_topSelection_cfg.py' $?

# cmsRun ${LOCAL_TEST_DIR}/patTuple_userData_cfg.py || die 'Failure using patTuple_userData_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/patTuple_metUncertainties_cfg.py || die 'Failure using patTuple_metUncertainties_cfg.py' $?

#---- disabled while the release is still open and changes to AOD event content are still allowed
#cmsRun ${LOCAL_TEST_DIR}/patMiniAOD_standard_cfg.py || die 'Failure using patMiniAOD_standard_cfg.py' $?

# Not needed in IBs
# cmsRun ${LOCAL_TEST_DIR}/patTuple_onlyElectrons_cfg.py || die 'Failure using patTuple_onlyElectrons_cfg.py' $?
# cmsRun ${LOCAL_TEST_DIR}/patTuple_onlyJets_cfg.py || die 'Failure using patTuple_onlyJets_cfg.py' $?
# cmsRun ${LOCAL_TEST_DIR}/patTuple_onlyMET_cfg.py || die 'Failure using patTuple_onlyMET_cfg.py' $?
# cmsRun ${LOCAL_TEST_DIR}/patTuple_onlyMuons_cfg.py || die 'Failure using patTuple_onlyMuons_cfg.py' $?
# cmsRun ${LOCAL_TEST_DIR}/patTuple_onlyPhotons_cfg.py || die 'Failure using patTuple_onlyPhotons_cfg.py' $?
# cmsRun ${LOCAL_TEST_DIR}/patTuple_onlyTaus_cfg.py || die 'Failure using patTuple_onlyTaus_cfg.py' $?

