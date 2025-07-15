#! /bin/bash
function die { echo $1: status $2 ; exit $2; }

echo "TESTING SagittaBiasNtuplizer Analyser with RECO input..."
cmsRun ${CMSSW_BASE}/src/Alignment/OfflineValidation/test/SagittaBiasNtuplizer_cfg.py || die "Failure running SagittaBiasNtuplizer_cfg.py (with RECO input)" $?

echo "TESTING SagittaBiasNtuplizer Analyser with ALCARECO input..."
cmsRun ${CMSSW_BASE}/src/Alignment/OfflineValidation/test/SagittaBiasNtuplizer_cfg.py globalTag=142X_mcRun3_2025_realistic_v7 fromRECO=False  myfile=/store/relval/CMSSW_15_1_0_pre1/RelValZMM_14/ALCARECO/TkAlDiMuonAndVertex-142X_mcRun3_2025_realistic_v7_STD_RegeneratedGS_2025_noPU-v1/2580000/b5e3fc09-7b77-42b8-9d69-d37e6fcfb5b8.root || die "Failure running SagittaBiasNtuplizer_cfg.py (with ALCARECO input)" $?
