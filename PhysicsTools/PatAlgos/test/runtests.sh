#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

for file in ${CMSSW_BASE}/src/PhysicsTools/PatAlgos/python/tools/*.py
do
    bn=`basename $file`
    if [ "$bn" != "__init__.py" ]; then
        python3 "$file" || die "unit tests for $bn failed" $?
    fi
done

cmsRun ${LOCAL_TEST_DIR}/patTuple_standard_cfg.py || die 'Failure using patTuple_standard_cfg.py' $?

# FIXME: event content broken in only available data RelVal
# cmsRun ${LOCAL_TEST_DIR}/patTuple_data_cfg.py || die 'Failure using patTuple_data_cfg.py' $? # Recent input RelVals currently not reachable (from CERN)

cmsRun ${LOCAL_TEST_DIR}/patTuple_PF2PAT_cfg.py || die 'Failure using patTuple_PF2PAT_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/patTuple_PATandPF2PAT_cfg.py || die 'Failure using patTuple_PATandPF2PAT_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/patTuple_addDecayInFlight_cfg.py || die 'Failure using patTuple_addDecayInFlight_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/patTuple_addBTagging_cfg.py || die 'Failure using patTuple_addBTagging_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/patTuple_addJets_cfg.py || die 'Failure using patTuple_addJets_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/patTuple_addTracks_cfg.py || die 'Failure using patTuple_addTracks_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/patTuple_addTriggerInfo_cfg.py || die 'Failure using patTuple_addTriggerInfo_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/patTuple_addVertexInfo_cfg.py || die 'Failure using patTuple_addVertexInfo_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/patTuple_fastsim_cfg.py || die 'Failure using patTuple_fastsim_cfg.py' $?

# cmsRun ${LOCAL_TEST_DIR}/patTuple_topSelection_cfg.py || die 'Failure using patTuple_topSelection_cfg.py' $?

# cmsRun ${LOCAL_TEST_DIR}/patTuple_userData_cfg.py || die 'Failure using patTuple_userData_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/patTuple_metUncertainties_cfg.py || die 'Failure using patTuple_metUncertainties_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/patTuple_updateMet_fromMiniAOD_cfg.py || die 'Failure using patTuple_updateMet_fromMiniAOD_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/patTuple_updateJets_fromMiniAOD_cfg.py || die 'Failure using patTuple_updateJets_fromMiniAOD_cfg.py' $?

#---- disabled while the release is still open and changes to AOD event content are still allowed
#cmsRun ${LOCAL_TEST_DIR}/patMiniAOD_standard_cfg.py || die 'Failure using patMiniAOD_standard_cfg.py' $?

# Not needed in IBs
# cmsRun ${LOCAL_TEST_DIR}/patTuple_onlyElectrons_cfg.py || die 'Failure using patTuple_onlyElectrons_cfg.py' $?
# cmsRun ${LOCAL_TEST_DIR}/patTuple_onlyJets_cfg.py || die 'Failure using patTuple_onlyJets_cfg.py' $?
# cmsRun ${LOCAL_TEST_DIR}/patTuple_onlyMET_cfg.py || die 'Failure using patTuple_onlyMET_cfg.py' $?
# cmsRun ${LOCAL_TEST_DIR}/patTuple_onlyMuons_cfg.py || die 'Failure using patTuple_onlyMuons_cfg.py' $?
# cmsRun ${LOCAL_TEST_DIR}/patTuple_onlyPhotons_cfg.py || die 'Failure using patTuple_onlyPhotons_cfg.py' $?
# cmsRun ${LOCAL_TEST_DIR}/patTuple_onlyTaus_cfg.py || die 'Failure using patTuple_onlyTaus_cfg.py' $?

