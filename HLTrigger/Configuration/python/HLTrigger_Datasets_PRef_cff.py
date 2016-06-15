# getDatasets.py

import FWCore.ParameterSet.Config as cms


# dump of the Stream PhysicsEGammaCommissioning Datasets defined in the HLT table as Stream A Datasets

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetEmptyBX_selector
streamA_datasetEmptyBX_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetEmptyBX_selector.l1tResults = cms.InputTag('')
streamA_datasetEmptyBX_selector.throw      = cms.bool(False)
streamA_datasetEmptyBX_selector.triggerConditions = cms.vstring('HLT_L1Tech5_BPTX_PlusOnly_v3', 
    'HLT_L1Tech6_BPTX_MinusOnly_v2', 
    'HLT_L1Tech7_NoBPTX_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHLTPhysics_selector
streamA_datasetHLTPhysics_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHLTPhysics_selector.l1tResults = cms.InputTag('')
streamA_datasetHLTPhysics_selector.throw      = cms.bool(False)
streamA_datasetHLTPhysics_selector.triggerConditions = cms.vstring('HLT_Physics_v3')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHighPtLowerPhotons_selector
streamA_datasetHighPtLowerPhotons_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHighPtLowerPhotons_selector.l1tResults = cms.InputTag('')
streamA_datasetHighPtLowerPhotons_selector.throw      = cms.bool(False)
streamA_datasetHighPtLowerPhotons_selector.triggerConditions = cms.vstring('HLT_HISinglePhoton10_Eta1p5ForPPRef_v2', 
    'HLT_HISinglePhoton10_Eta3p1ForPPRef_v2', 
    'HLT_HISinglePhoton15_Eta1p5ForPPRef_v2', 
    'HLT_HISinglePhoton15_Eta3p1ForPPRef_v2', 
    'HLT_HISinglePhoton20_Eta1p5ForPPRef_v2', 
    'HLT_HISinglePhoton20_Eta3p1ForPPRef_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHighPtPhoton30AndZ_selector
streamA_datasetHighPtPhoton30AndZ_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHighPtPhoton30AndZ_selector.l1tResults = cms.InputTag('')
streamA_datasetHighPtPhoton30AndZ_selector.throw      = cms.bool(False)
streamA_datasetHighPtPhoton30AndZ_selector.triggerConditions = cms.vstring('HLT_HIDoublePhoton15_Eta1p5_Mass50_1000ForPPRef_v2', 
    'HLT_HIDoublePhoton15_Eta1p5_Mass50_1000_R9HECutForPPRef_v2', 
    'HLT_HIDoublePhoton15_Eta2p1_Mass50_1000_R9CutForPPRef_v2', 
    'HLT_HIDoublePhoton15_Eta2p5_Mass50_1000_R9SigmaHECutForPPRef_v2', 
    'HLT_HISinglePhoton30_Eta1p5ForPPRef_v2', 
    'HLT_HISinglePhoton30_Eta3p1ForPPRef_v2', 
    'HLT_HISinglePhoton40_Eta1p5ForPPRef_v2', 
    'HLT_HISinglePhoton40_Eta3p1ForPPRef_v2', 
    'HLT_HISinglePhoton50_Eta1p5ForPPRef_v2', 
    'HLT_HISinglePhoton50_Eta3p1ForPPRef_v2', 
    'HLT_HISinglePhoton60_Eta1p5ForPPRef_v2', 
    'HLT_HISinglePhoton60_Eta3p1ForPPRef_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetL1MinimumBias_selector
streamA_datasetL1MinimumBias_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetL1MinimumBias_selector.l1tResults = cms.InputTag('')
streamA_datasetL1MinimumBias_selector.throw      = cms.bool(False)
streamA_datasetL1MinimumBias_selector.triggerConditions = cms.vstring('HLT_L1MinimumBiasHF1AND_v2', 
    'HLT_L1MinimumBiasHF1OR_v2', 
    'HLT_L1MinimumBiasHF2AND_v2', 
    'HLT_L1MinimumBiasHF2ORNoBptxGating_v2', 
    'HLT_L1MinimumBiasHF2OR_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetTOTEM_minBias_selector
streamA_datasetTOTEM_minBias_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetTOTEM_minBias_selector.l1tResults = cms.InputTag('')
streamA_datasetTOTEM_minBias_selector.throw      = cms.bool(False)
streamA_datasetTOTEM_minBias_selector.triggerConditions = cms.vstring('HLT_L1TOTEM1_MinBias_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetTOTEM_zeroBias_selector
streamA_datasetTOTEM_zeroBias_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetTOTEM_zeroBias_selector.l1tResults = cms.InputTag('')
streamA_datasetTOTEM_zeroBias_selector.throw      = cms.bool(False)
streamA_datasetTOTEM_zeroBias_selector.triggerConditions = cms.vstring('HLT_L1TOTEM2_ZeroBias_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetZeroBias_selector
streamA_datasetZeroBias_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetZeroBias_selector.l1tResults = cms.InputTag('')
streamA_datasetZeroBias_selector.throw      = cms.bool(False)
streamA_datasetZeroBias_selector.triggerConditions = cms.vstring('HLT_Random_v2', 
    'HLT_ZeroBias_v3')


# dump of the Stream PhysicsHadronsTaus Datasets defined in the HLT table as Stream A Datasets

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetBTagCSV_selector
streamA_datasetBTagCSV_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetBTagCSV_selector.l1tResults = cms.InputTag('')
streamA_datasetBTagCSV_selector.throw      = cms.bool(False)
streamA_datasetBTagCSV_selector.triggerConditions = cms.vstring('HLT_AK4PFBJetBCSV60_Eta2p1ForPPRef_v2', 
    'HLT_AK4PFBJetBCSV80_Eta2p1ForPPRef_v2', 
    'HLT_AK4PFBJetBSSV60_Eta2p1ForPPRef_v2', 
    'HLT_AK4PFBJetBSSV80_Eta2p1ForPPRef_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHeavyFlavor_selector
streamA_datasetHeavyFlavor_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHeavyFlavor_selector.l1tResults = cms.InputTag('')
streamA_datasetHeavyFlavor_selector.throw      = cms.bool(False)
streamA_datasetHeavyFlavor_selector.triggerConditions = cms.vstring('HLT_DmesonPPTrackingGlobal_Dpt15ForPPRef_v2', 
    'HLT_DmesonPPTrackingGlobal_Dpt20ForPPRef_v2', 
    'HLT_DmesonPPTrackingGlobal_Dpt30ForPPRef_v2', 
    'HLT_DmesonPPTrackingGlobal_Dpt40ForPPRef_v2', 
    'HLT_DmesonPPTrackingGlobal_Dpt50ForPPRef_v2', 
    'HLT_DmesonPPTrackingGlobal_Dpt60ForPPRef_v2', 
    'HLT_DmesonPPTrackingGlobal_Dpt8ForPPRef_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHighPtJet80_selector
streamA_datasetHighPtJet80_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHighPtJet80_selector.l1tResults = cms.InputTag('')
streamA_datasetHighPtJet80_selector.throw      = cms.bool(False)
streamA_datasetHighPtJet80_selector.triggerConditions = cms.vstring('HLT_AK4CaloJet100_Eta5p1ForPPRef_v2', 
    'HLT_AK4CaloJet100_Jet35_Eta0p7ForPPRef_v2', 
    'HLT_AK4CaloJet100_Jet35_Eta1p1ForPPRef_v2', 
    'HLT_AK4CaloJet110_Eta5p1ForPPRef_v2', 
    'HLT_AK4CaloJet120_Eta5p1ForPPRef_v2', 
    'HLT_AK4CaloJet150ForPPRef_v2', 
    'HLT_AK4CaloJet80_45_45_Eta2p1ForPPRef_v2', 
    'HLT_AK4CaloJet80_Eta5p1ForPPRef_v2', 
    'HLT_AK4CaloJet80_Jet35_Eta0p7ForPPRef_v2', 
    'HLT_AK4CaloJet80_Jet35_Eta1p1ForPPRef_v2', 
    'HLT_AK4PFJet100_Eta5p1ForPPRef_v2', 
    'HLT_AK4PFJet110_Eta5p1ForPPRef_v2', 
    'HLT_AK4PFJet120_Eta5p1ForPPRef_v2', 
    'HLT_AK4PFJet80_Eta5p1ForPPRef_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetHighPtLowerJets_selector
streamA_datasetHighPtLowerJets_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetHighPtLowerJets_selector.l1tResults = cms.InputTag('')
streamA_datasetHighPtLowerJets_selector.throw      = cms.bool(False)
streamA_datasetHighPtLowerJets_selector.triggerConditions = cms.vstring('HLT_AK4CaloJet40_Eta5p1ForPPRef_v2', 
    'HLT_AK4CaloJet60_Eta5p1ForPPRef_v2', 
    'HLT_AK4PFJet40_Eta5p1ForPPRef_v2', 
    'HLT_AK4PFJet60_Eta5p1ForPPRef_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetJetHT_selector
streamA_datasetJetHT_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetJetHT_selector.l1tResults = cms.InputTag('')
streamA_datasetJetHT_selector.throw      = cms.bool(False)
streamA_datasetJetHT_selector.triggerConditions = cms.vstring('HLT_AK4PFDJet60_Eta2p1ForPPRef_v2', 
    'HLT_AK4PFDJet80_Eta2p1ForPPRef_v2')


# dump of the Stream PhysicsMuons Datasets defined in the HLT table as Stream A Datasets

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetDoubleMuon_selector
streamA_datasetDoubleMuon_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetDoubleMuon_selector.l1tResults = cms.InputTag('')
streamA_datasetDoubleMuon_selector.throw      = cms.bool(False)
streamA_datasetDoubleMuon_selector.triggerConditions = cms.vstring('HLT_HIL1DoubleMu0ForPPRef_v2', 
    'HLT_HIL1DoubleMu10ForPPRef_v2', 
    'HLT_HIL2DoubleMu0_NHitQForPPRef_v2', 
    'HLT_HIL3DoubleMu0_OS_m2p5to4p5ForPPRef_v2', 
    'HLT_HIL3DoubleMu0_OS_m7to14ForPPRef_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetMuPlusX_selector
streamA_datasetMuPlusX_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetMuPlusX_selector.l1tResults = cms.InputTag('')
streamA_datasetMuPlusX_selector.throw      = cms.bool(False)
streamA_datasetMuPlusX_selector.triggerConditions = cms.vstring('HLT_HIL2Mu3Eta2p5_AK4CaloJet100Eta2p1ForPPRef_v2', 
    'HLT_HIL2Mu3Eta2p5_AK4CaloJet40Eta2p1ForPPRef_v2', 
    'HLT_HIL2Mu3Eta2p5_AK4CaloJet60Eta2p1ForPPRef_v2', 
    'HLT_HIL2Mu3Eta2p5_AK4CaloJet80Eta2p1ForPPRef_v2', 
    'HLT_HIL2Mu3Eta2p5_HIPhoton10Eta1p5ForPPRef_v2', 
    'HLT_HIL2Mu3Eta2p5_HIPhoton15Eta1p5ForPPRef_v2', 
    'HLT_HIL2Mu3Eta2p5_HIPhoton20Eta1p5ForPPRef_v2', 
    'HLT_HIL2Mu3Eta2p5_HIPhoton30Eta1p5ForPPRef_v2', 
    'HLT_HIL2Mu3Eta2p5_HIPhoton40Eta1p5ForPPRef_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetSingleMuHighPt_selector
streamA_datasetSingleMuHighPt_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetSingleMuHighPt_selector.l1tResults = cms.InputTag('')
streamA_datasetSingleMuHighPt_selector.throw      = cms.bool(False)
streamA_datasetSingleMuHighPt_selector.triggerConditions = cms.vstring('HLT_HIL2Mu15ForPPRef_v2', 
    'HLT_HIL2Mu20ForPPRef_v2', 
    'HLT_HIL2Mu5_NHitQ10ForPPRef_v2', 
    'HLT_HIL2Mu7_NHitQ10ForPPRef_v2', 
    'HLT_HIL3Mu15ForPPRef_v2', 
    'HLT_HIL3Mu20ForPPRef_v2', 
    'HLT_HIL3Mu5_NHitQ15ForPPRef_v2', 
    'HLT_HIL3Mu7_NHitQ15ForPPRef_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamA_datasetSingleMuLowPt_selector
streamA_datasetSingleMuLowPt_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamA_datasetSingleMuLowPt_selector.l1tResults = cms.InputTag('')
streamA_datasetSingleMuLowPt_selector.throw      = cms.bool(False)
streamA_datasetSingleMuLowPt_selector.triggerConditions = cms.vstring('HLT_HIL2Mu3_NHitQ10ForPPRef_v2', 
    'HLT_HIL3Mu3_NHitQ15ForPPRef_v2')

