#!/bin/sh

function die { echo $1: status $2 ; exit $2; }

echo -e "TESTING step1 with RECO inputs ...\n\n"
cmsRun ${SCRAM_TEST_PATH}/testTrackResolution_cfg.py isUnitTest=True || die "Failure running testTrackResolution_cfg.py isUnitTest=True" $?

echo -e "TESTING step1 with ALCARECO inputs ...\n\n"
cmsRun ${SCRAM_TEST_PATH}/testTrackResolution_cfg.py isUnitTest=True isAlCaReco=True inputFile=/store/mc/Run3Winter23Reco/DYJetsToMuMu_M-50_TuneCP5_13p6TeV-madgraphMLM-pythia8/ALCARECO/TkAlDiMuonAndVertex-TRKDesignNoPU_AlcaRecoTRKMu_designGaussSigmaZ4cm_125X_mcRun3_2022_design_v6-v1/60000/93401af5-0da6-40ce-82e4-d5571c93dd97.root || die "Failure running testTrackResolution_cfg.py isUnitTest=True isAlCaReco=True" $?

echo -e "TESTING harvesting with RECO inputs ...\n\n"
cmsRun ${SCRAM_TEST_PATH}/testTrackResolutionHarvesting_cfg.py || die "Failure running testTrackResolutionHarvesting_cfg.py" $?

echo -e "TESTING harvesting with ALCARECO inputs ...\n\n"
cmsRun ${SCRAM_TEST_PATH}/testTrackResolutionHarvesting_cfg.py inputFile=step1_DQM_LayerRot_9p43e-6_fromALCA.root || die "Failure running testTrackResolutionHarvesting_cfg.py inputFile=step1_DQM_LayerRot_9p43e-6_fromALCA.root" $?
