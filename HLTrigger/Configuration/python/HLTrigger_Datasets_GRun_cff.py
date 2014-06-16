# getDatasets.py

import FWCore.ParameterSet.Config as cms


# dump of the Stream A Datasets defined in the HLT table as Stream A Datasets

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetInitialPD_selector
streamA_datasetInitialPD_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetInitialPD_selector.l1tResults = cms.InputTag('')
streamA_datasetInitialPD_selector.throw      = cms.bool(False)
streamA_datasetInitialPD_selector.triggerConditions = cms.vstring('HLT_BTagCSV07_v1', 
    'HLT_CaloJet260_v1', 
    'HLT_DoubleEle33_CaloIdL_GsfTrkIdVL_v1', 
    'HLT_DoubleMediumIsoPFTau35_Trk1_eta2p1_Prong1_Reg_v1', 
    'HLT_DoubleMediumIsoPFTau35_Trk1_eta2p1_Prong1_v1', 
    'HLT_DoubleMediumIsoPFTau35_Trk1_eta2p1_Reg_v1', 
    'HLT_DoubleMediumIsoPFTau35_Trk1_eta2p1_v1', 
    'HLT_DoubleMu4NoFilters_Jpsi_Displaced_v1', 
    'HLT_Ele17_Ele8_Gsf_v1', 
    'HLT_Ele22_eta2p1_WP90Rho_LooseIsoPFTau20_v1', 
    'HLT_Ele27_WP80_Gsf_v1', 
    'HLT_HT650_v1', 
    'HLT_IsoMu17_eta2p1_LooseIsoPFTau20_v1', 
    'HLT_IsoMu24_IterTrk02_v1', 
    'HLT_IsoTkMu24_IterTrk02_v1', 
    'HLT_IterativeTracking_v1', 
    'HLT_JetE50_NoBPTX3BX_NoHalo_v1', 
    'HLT_LooseIsoPFTau35_Trk20_Prong1_MET70_v1', 
    'HLT_Mu17_Mu8_v1', 
    'HLT_Mu17_NoFilters_v1', 
    'HLT_Mu17_TkMu8_v1', 
    'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_v1', 
    'HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_v1', 
    'HLT_Mu40_v1', 
    'HLT_PFHT650_v1', 
    'HLT_PFJet260_v1', 
    'HLT_PFJet40_v1', 
    'HLT_PFMET180_NoiseCleaned_v1', 
    'HLT_PFNoPUHT650_v1', 
    'HLT_PFNoPUJet260_v1', 
    'HLT_PFchMET90_NoiseCleaned_v1', 
    'HLT_Photon20_CaloIdVL_IsoL_v1', 
    'HLT_Photon26_R9Id85_Photon18_CaloId10_Iso50_v1', 
    'HLT_ReducedIterativeTracking_v1')

