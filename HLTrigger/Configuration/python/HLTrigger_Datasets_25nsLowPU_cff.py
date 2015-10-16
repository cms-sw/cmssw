# getDatasets.py

import FWCore.ParameterSet.Config as cms


# dump of the Stream PhysicsEGammaCommissioning Datasets defined in the HLT table as Stream A Datasets

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetCastorJets_selector
streamA_datasetCastorJets_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetCastorJets_selector.l1tResults = cms.InputTag('')
streamA_datasetCastorJets_selector.throw      = cms.bool(False)
streamA_datasetCastorJets_selector.triggerConditions = cms.vstring()

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetCommissioning_selector
streamA_datasetCommissioning_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetCommissioning_selector.l1tResults = cms.InputTag('')
streamA_datasetCommissioning_selector.throw      = cms.bool(False)
streamA_datasetCommissioning_selector.triggerConditions = cms.vstring('HLT_Activity_Ecal_SC7_v1', 
    'HLT_IsoTrackHB_v1', 
    'HLT_IsoTrackHE_v1', 
    'HLT_L1SingleMuOpen_DT_v1', 
    'HLT_L1Tech_DT_GlobalOR_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetDoubleEG_selector
streamA_datasetDoubleEG_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetDoubleEG_selector.l1tResults = cms.InputTag('')
streamA_datasetDoubleEG_selector.throw      = cms.bool(False)
streamA_datasetDoubleEG_selector.triggerConditions = cms.vstring('HLT_Ele5_SC5_JPsi_Mass2to4p5_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetEGMLowPU_selector
streamA_datasetEGMLowPU_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetEGMLowPU_selector.l1tResults = cms.InputTag('')
streamA_datasetEGMLowPU_selector.throw      = cms.bool(False)
streamA_datasetEGMLowPU_selector.triggerConditions = cms.vstring('HLT_Activity_Ecal_SC7_v1', 
    'HLT_Ele5_SC5_JPsi_Mass2to4p5_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetEmptyBX_selector
streamA_datasetEmptyBX_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetEmptyBX_selector.l1tResults = cms.InputTag('')
streamA_datasetEmptyBX_selector.throw      = cms.bool(False)
streamA_datasetEmptyBX_selector.triggerConditions = cms.vstring('HLT_L1Tech5_BPTX_PlusOnly_v1', 
    'HLT_L1Tech6_BPTX_MinusOnly_v1', 
    'HLT_L1Tech7_NoBPTX_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetFSQJets1_selector
streamA_datasetFSQJets1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetFSQJets1_selector.l1tResults = cms.InputTag('')
streamA_datasetFSQJets1_selector.throw      = cms.bool(False)
streamA_datasetFSQJets1_selector.triggerConditions = cms.vstring('HLT_PFJet15_NoCaloMatched_v2', 
    'HLT_PFJet25_NoCaloMatched_v2', 
    'HLT_PFJet40_NoCaloMatched_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetFSQJets2_selector
streamA_datasetFSQJets2_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetFSQJets2_selector.l1tResults = cms.InputTag('')
streamA_datasetFSQJets2_selector.throw      = cms.bool(False)
streamA_datasetFSQJets2_selector.triggerConditions = cms.vstring('HLT_DiPFJetAve15_Central_v2', 
    'HLT_DiPFJetAve15_HFJEC_v2', 
    'HLT_DiPFJetAve25_Central_v2', 
    'HLT_DiPFJetAve25_HFJEC_v2', 
    'HLT_DiPFJetAve35_Central_v2', 
    'HLT_DiPFJetAve35_HFJEC_v2', 
    'HLT_PFJet15_FwdEta2_NoCaloMatched_v2', 
    'HLT_PFJet15_FwdEta3_NoCaloMatched_v2', 
    'HLT_PFJet25_FwdEta2_NoCaloMatched_v2', 
    'HLT_PFJet25_FwdEta3_NoCaloMatched_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetFSQJets3_selector
streamA_datasetFSQJets3_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetFSQJets3_selector.l1tResults = cms.InputTag('')
streamA_datasetFSQJets3_selector.throw      = cms.bool(False)
streamA_datasetFSQJets3_selector.triggerConditions = cms.vstring('HLT_DiPFJet15_FBEta2_NoCaloMatched_v2', 
    'HLT_DiPFJet15_FBEta3_NoCaloMatched_v2', 
    'HLT_DiPFJet15_NoCaloMatched_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetFullTrack_selector
streamA_datasetFullTrack_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetFullTrack_selector.l1tResults = cms.InputTag('')
streamA_datasetFullTrack_selector.throw      = cms.bool(False)
streamA_datasetFullTrack_selector.triggerConditions = cms.vstring('HLT_FullTrack12_v2', 
    'HLT_FullTrack20_v2', 
    'HLT_FullTrack30_v2', 
    'HLT_FullTrack50_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHINCaloJet40_selector
streamA_datasetHINCaloJet40_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHINCaloJet40_selector.l1tResults = cms.InputTag('')
streamA_datasetHINCaloJet40_selector.throw      = cms.bool(False)
streamA_datasetHINCaloJet40_selector.triggerConditions = cms.vstring('HLT_AK4CaloJet40_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHINCaloJetsOther_selector
streamA_datasetHINCaloJetsOther_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHINCaloJetsOther_selector.l1tResults = cms.InputTag('')
streamA_datasetHINCaloJetsOther_selector.throw      = cms.bool(False)
streamA_datasetHINCaloJetsOther_selector.triggerConditions = cms.vstring('HLT_AK4CaloJet100_v2', 
    'HLT_AK4CaloJet30_v2', 
    'HLT_AK4CaloJet50_v2', 
    'HLT_AK4CaloJet80_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHINMuon_selector
streamA_datasetHINMuon_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHINMuon_selector.l1tResults = cms.InputTag('')
streamA_datasetHINMuon_selector.throw      = cms.bool(False)
streamA_datasetHINMuon_selector.triggerConditions = cms.vstring('HLT_HIL1DoubleMu0_v1', 
    'HLT_HIL2DoubleMu0_v2', 
    'HLT_HIL2Mu3_v2', 
    'HLT_HIL3Mu3_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHINPFJetsOther_selector
streamA_datasetHINPFJetsOther_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHINPFJetsOther_selector.l1tResults = cms.InputTag('')
streamA_datasetHINPFJetsOther_selector.throw      = cms.bool(False)
streamA_datasetHINPFJetsOther_selector.triggerConditions = cms.vstring('HLT_AK4PFJet100_v2', 
    'HLT_AK4PFJet30_v2', 
    'HLT_AK4PFJet50_v2', 
    'HLT_AK4PFJet80_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHINPhoton_selector
streamA_datasetHINPhoton_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHINPhoton_selector.l1tResults = cms.InputTag('')
streamA_datasetHINPhoton_selector.throw      = cms.bool(False)
streamA_datasetHINPhoton_selector.triggerConditions = cms.vstring('HLT_HISinglePhoton10_v2', 
    'HLT_HISinglePhoton15_v2', 
    'HLT_HISinglePhoton20_v2', 
    'HLT_HISinglePhoton40_v2', 
    'HLT_HISinglePhoton60_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHLTPhysics_selector
streamA_datasetHLTPhysics_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHLTPhysics_selector.l1tResults = cms.InputTag('')
streamA_datasetHLTPhysics_selector.throw      = cms.bool(False)
streamA_datasetHLTPhysics_selector.triggerConditions = cms.vstring('HLT_Physics_v1')

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

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHighMultiplicity_selector
streamA_datasetHighMultiplicity_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHighMultiplicity_selector.l1tResults = cms.InputTag('')
streamA_datasetHighMultiplicity_selector.throw      = cms.bool(False)
streamA_datasetHighMultiplicity_selector.triggerConditions = cms.vstring('HLT_PixelTracks_Multiplicity110_v2', 
    'HLT_PixelTracks_Multiplicity135_v2', 
    'HLT_PixelTracks_Multiplicity160_v2', 
    'HLT_PixelTracks_Multiplicity60_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHighMultiplicity85_selector
streamA_datasetHighMultiplicity85_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHighMultiplicity85_selector.l1tResults = cms.InputTag('')
streamA_datasetHighMultiplicity85_selector.throw      = cms.bool(False)
streamA_datasetHighMultiplicity85_selector.triggerConditions = cms.vstring('HLT_PixelTracks_Multiplicity85_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetL1MinimumBias_selector
streamA_datasetL1MinimumBias_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetL1MinimumBias_selector.l1tResults = cms.InputTag('')
streamA_datasetL1MinimumBias_selector.throw      = cms.bool(False)
streamA_datasetL1MinimumBias_selector.triggerConditions = cms.vstring('HLT_L1MinimumBiasHF1AND_v1', 
    'HLT_L1MinimumBiasHF1OR_v1', 
    'HLT_L1MinimumBiasHF2AND_v1', 
    'HLT_L1MinimumBiasHF2OR_NoBptxGate_v1', 
    'HLT_L1MinimumBiasHF2OR_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetNoBPTX_selector
streamA_datasetNoBPTX_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetNoBPTX_selector.l1tResults = cms.InputTag('')
streamA_datasetNoBPTX_selector.throw      = cms.bool(False)
streamA_datasetNoBPTX_selector.triggerConditions = cms.vstring('HLT_JetE30_NoBPTX3BX_NoHalo_v2', 
    'HLT_JetE30_NoBPTX_v2', 
    'HLT_JetE50_NoBPTX3BX_NoHalo_v2', 
    'HLT_JetE70_NoBPTX3BX_NoHalo_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetTOTEM_minBias_selector
streamA_datasetTOTEM_minBias_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetTOTEM_minBias_selector.l1tResults = cms.InputTag('')
streamA_datasetTOTEM_minBias_selector.throw      = cms.bool(False)
streamA_datasetTOTEM_minBias_selector.triggerConditions = cms.vstring('HLT_L1TOTEM1_MinBias_v1', 
    'HLT_L1TOTEM2_ZeroBias_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetTOTEM_romanPots_selector
streamA_datasetTOTEM_romanPots_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetTOTEM_romanPots_selector.l1tResults = cms.InputTag('')
streamA_datasetTOTEM_romanPots_selector.throw      = cms.bool(False)
streamA_datasetTOTEM_romanPots_selector.triggerConditions = cms.vstring('HLT_L1RomanPots_SinglePixelTrack02_v2', 
    'HLT_L1RomanPots_SinglePixelTrack04_v2', 
    'HLT_L1TOTEM0_RomanPotsAND_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetToTOTEM_selector
streamA_datasetToTOTEM_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetToTOTEM_selector.l1tResults = cms.InputTag('')
streamA_datasetToTOTEM_selector.throw      = cms.bool(False)
streamA_datasetToTOTEM_selector.triggerConditions = cms.vstring('HLT_L1DoubleJet20_v1', 
    'HLT_L1DoubleJet28_v1', 
    'HLT_L1DoubleJet32_v1', 
    'HLT_L1DoubleMuOpen_v1', 
    'HLT_L1SingleMuOpen_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetZeroBias_selector
streamA_datasetZeroBias_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetZeroBias_selector.l1tResults = cms.InputTag('')
streamA_datasetZeroBias_selector.throw      = cms.bool(False)
streamA_datasetZeroBias_selector.triggerConditions = cms.vstring('HLT_Random_v1', 
    'HLT_ZeroBias_v1')

