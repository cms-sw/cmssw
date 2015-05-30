# getDatasets.py

import FWCore.ParameterSet.Config as cms


# dump of the Stream A Datasets defined in the HLT table as Stream A Datasets

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHcalHPDNoise_selector
streamA_datasetHcalHPDNoise_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHcalHPDNoise_selector.l1tResults = cms.InputTag('')
streamA_datasetHcalHPDNoise_selector.throw      = cms.bool(False)
streamA_datasetHcalHPDNoise_selector.triggerConditions = cms.vstring('HLT_GlobalRunHPDNoise_v1', 
    'HLT_L1Tech_HBHEHO_totalOR_v1', 
    'HLT_L1Tech_HCAL_HF_single_channel_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHcalNZS_selector
streamA_datasetHcalNZS_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHcalNZS_selector.l1tResults = cms.InputTag('')
streamA_datasetHcalNZS_selector.throw      = cms.bool(False)
streamA_datasetHcalNZS_selector.triggerConditions = cms.vstring('HLT_HcalNZS_v1', 
    'HLT_HcalPhiSym_v1', 
    'HLT_HcalUTCA_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetInitialPD_selector
streamA_datasetInitialPD_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetInitialPD_selector.l1tResults = cms.InputTag('')
streamA_datasetInitialPD_selector.throw      = cms.bool(False)
streamA_datasetInitialPD_selector.triggerConditions = cms.vstring('HLT_AK4CaloJet100_v2', 
    'HLT_AK4CaloJet30_v2', 
    'HLT_AK4CaloJet40_v2', 
    'HLT_AK4CaloJet50_v2', 
    'HLT_AK4CaloJet80_v2', 
    'HLT_AK4PFJet100_v2', 
    'HLT_AK4PFJet30_v2', 
    'HLT_AK4PFJet50_v2', 
    'HLT_AK4PFJet80_v2', 
    'HLT_Activity_Ecal_SC7_v1', 
    'HLT_DiPFJet15_FBEta2_NoCaloMatched_v2', 
    'HLT_DiPFJet15_FBEta3_NoCaloMatched_v2', 
    'HLT_DiPFJet15_NoCaloMatched_v2', 
    'HLT_DiPFJetAve15_Central_v2', 
    'HLT_DiPFJetAve15_HFJEC_v2', 
    'HLT_DiPFJetAve25_Central_v2', 
    'HLT_DiPFJetAve25_HFJEC_v2', 
    'HLT_DiPFJetAve30_HFJEC_v2', 
    'HLT_DiPFJetAve35_Central_v2', 
    'HLT_DiPFJetAve35_HFJEC_v2', 
    'HLT_Ele5_SC5_JPsi_Mass2to4p5_v2', 
    'HLT_FullTrack12_v2', 
    'HLT_FullTrack20_v2', 
    'HLT_FullTrack30_v2', 
    'HLT_FullTrack50_v2', 
    'HLT_HISinglePhoton10_v2', 
    'HLT_HISinglePhoton15_v2', 
    'HLT_HISinglePhoton20_v2', 
    'HLT_HISinglePhoton40_v2', 
    'HLT_HISinglePhoton60_v2', 
    'HLT_IsoTrackHB_v1', 
    'HLT_IsoTrackHE_v1', 
    'HLT_L1CastorHighJet_v1', 
    'HLT_L1CastorMediumJet_v1', 
    'HLT_L1CastorMuon_v1', 
    'HLT_L1DoubleJet20_v1', 
    'HLT_L1DoubleJet28_v1', 
    'HLT_L1DoubleJet32_v1', 
    'HLT_L1DoubleMuOpen_v1', 
    'HLT_L1MinimumBiasHF1AND_v1', 
    'HLT_L1MinimumBiasHF1OR_v1', 
    'HLT_L1MinimumBiasHF2AND_v1', 
    'HLT_L1MinimumBiasHF2OR_v1', 
    'HLT_L1RomanPots_SinglePixelTrack02_v2', 
    'HLT_L1RomanPots_SinglePixelTrack04_v2', 
    'HLT_L1RomanPots_v1', 
    'HLT_L1SingleMuOpen_DT_v1', 
    'HLT_L1SingleMuOpen_v1', 
    'HLT_L1TOTEM0_RomanPotsAND_v1', 
    'HLT_L1TOTEM1_MinBias_v1', 
    'HLT_L1TOTEM3_ZeroBias_v1', 
    'HLT_L1Tech5_BPTX_PlusOnly_v1', 
    'HLT_L1Tech62_CASTORJet_PFJet15_v1', 
    'HLT_L1Tech62_CASTORJet_v1', 
    'HLT_L1Tech63_CASTORHaloMuon_v2', 
    'HLT_L1Tech6_BPTX_MinusOnly_v1', 
    'HLT_L1Tech7_NoBPTX_v1', 
    'HLT_L1Tech_DT_GlobalOR_v1', 
    'HLT_PFJet15_FwdEta2_NoCaloMatched_v2', 
    'HLT_PFJet15_FwdEta3_NoCaloMatched_v2', 
    'HLT_PFJet15_NoCaloMatched_v2', 
    'HLT_PFJet20_NoCaloMatched_v2', 
    'HLT_PFJet25_FwdEta2_NoCaloMatched_v2', 
    'HLT_PFJet25_FwdEta3_NoCaloMatched_v2', 
    'HLT_PFJet25_NoCaloMatched_v2', 
    'HLT_PFJet40_FwdEta2_NoCaloMatched_v2', 
    'HLT_PFJet40_FwdEta3_NoCaloMatched_v2', 
    'HLT_PFJet40_NoCaloMatched_v2', 
    'HLT_Physics_v1', 
    'HLT_PixelTracks_Multiplicity110_v2', 
    'HLT_PixelTracks_Multiplicity135_v2', 
    'HLT_PixelTracks_Multiplicity160_v2', 
    'HLT_PixelTracks_Multiplicity60_v2', 
    'HLT_PixelTracks_Multiplicity85_v2', 
    'HLT_Random_v1', 
    'HLT_ZeroBias_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetInitialPDForHI_selector
streamA_datasetInitialPDForHI_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetInitialPDForHI_selector.l1tResults = cms.InputTag('')
streamA_datasetInitialPDForHI_selector.throw      = cms.bool(False)
streamA_datasetInitialPDForHI_selector.triggerConditions = cms.vstring('HLT_HIL1DoubleMu0_v1', 
    'HLT_HIL2DoubleMu0_v2', 
    'HLT_HIL2Mu3_v2', 
    'HLT_HIL3Mu3_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMinimumBias_selector
streamA_datasetMinimumBias_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMinimumBias_selector.l1tResults = cms.InputTag('')
streamA_datasetMinimumBias_selector.throw      = cms.bool(False)
streamA_datasetMinimumBias_selector.triggerConditions = cms.vstring('HLT_Activity_Ecal_SC7_v1', 
    'HLT_ZeroBias_v1')

