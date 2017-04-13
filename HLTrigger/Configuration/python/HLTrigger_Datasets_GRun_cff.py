# /dev/CMSSW_9_0_1/GRun

import FWCore.ParameterSet.Config as cms


# stream PhysicsCommissioning

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetHLTPhysics_selector
streamPhysicsCommissioning_datasetHLTPhysics_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetHLTPhysics_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetHLTPhysics_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetHLTPhysics_selector.triggerConditions = cms.vstring('HLT_Physics_v5')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetMonteCarlo_selector
streamPhysicsCommissioning_datasetMonteCarlo_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetMonteCarlo_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetMonteCarlo_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetMonteCarlo_selector.triggerConditions = cms.vstring('MC_AK4CaloJets_v3', 
    'MC_AK4PFJets_v6', 
    'MC_AK8CaloHT_v3', 
    'MC_AK8PFHT_v6', 
    'MC_AK8PFJets_v6', 
    'MC_AK8TrimPFJets_v6', 
    'MC_CaloHT_v3', 
    'MC_CaloMET_JetIdCleaned_v3', 
    'MC_CaloMET_v3', 
    'MC_CaloMHT_v3', 
    'MC_Diphoton10_10_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass10_v6', 
    'MC_DoubleEle5_CaloIdL_GsfTrkIdVL_MW_v6', 
    'MC_DoubleGlbTrkMu_TrkIsoVVL_DZ_v4', 
    'MC_DoubleL1Tau_MediumIsoPFTau32_Trk1_eta2p1_Reg_v6', 
    'MC_DoubleMuNoFiltersNoVtx_v3', 
    'MC_DoubleMu_TrkIsoVVL_DZ_v4', 
    'MC_Ele15_Ele10_CaloIdL_TrackIdL_IsoVL_DZ_v7', 
    'MC_Ele5_WPLoose_Gsf_v8', 
    'MC_IsoMu_v7', 
    'MC_IsoTkMu15_v6', 
    'MC_LooseIsoPFTau20_v5', 
    'MC_LooseIsoPFTau50_Trk30_eta2p1_v4', 
    'MC_PFHT_v6', 
    'MC_PFMET_v6', 
    'MC_PFMHT_v6', 
    'MC_ReducedIterativeTracking_v3')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetZeroBias_selector
streamPhysicsCommissioning_datasetZeroBias_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetZeroBias_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetZeroBias_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetZeroBias_selector.triggerConditions = cms.vstring('HLT_ZeroBias_v4')

