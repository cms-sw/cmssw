#! /bin/bash
function die { echo $1: status $2 ; exit $2; }

echo "TESTING SagittaBiasNtuplizer Analyser with RECO input..."
cmsRun ${CMSSW_BASE}/src/Alignment/OfflineValidation/test/SagittaBiasNtuplizer_cfg.py || die "Failure running SagittaBiasNtuplizer_cfg.py (with RECO input)" $?

echo "TESTING SagittaBiasNtuplizer Analyser with ALCARECO input..."
cmsRun ${CMSSW_BASE}/src/Alignment/OfflineValidation/test/SagittaBiasNtuplizer_cfg.py globalTag="auto:phase2_realistic" fromRECO=False  myfile=/store/relval/CMSSW_20_0_0_pre1/RelValZMM_14/ALCARECO/TkAlDiMuonAndVertex-150X_mcRun4_realistic_v1_STD_RegeneratedGS_D121_noPU-v1/2590000/b245c25c-c9cd-4329-b081-38a95f3b6bbe.root || die "Failure running SagittaBiasNtuplizer_cfg.py (with ALCARECO input)" $?
