# /dev/CMSSW_9_2_0/PRef

import FWCore.ParameterSet.Config as cms


# stream PhysicsCommissioning

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetHLTPhysics_selector
streamPhysicsCommissioning_datasetHLTPhysics_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetHLTPhysics_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetHLTPhysics_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetHLTPhysics_selector.triggerConditions = cms.vstring('HLT_Physics_v7')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetHighPtLowerPhotons_selector
streamPhysicsCommissioning_datasetHighPtLowerPhotons_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetHighPtLowerPhotons_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetHighPtLowerPhotons_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetHighPtLowerPhotons_selector.triggerConditions = cms.vstring('HLT_HISinglePhoton10_Eta1p5ForPPRef_v8', 
    'HLT_HISinglePhoton15_Eta1p5ForPPRef_v8', 
    'HLT_HISinglePhoton20_Eta1p5ForPPRef_v8')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetHighPtPhoton30AndZ_selector
streamPhysicsCommissioning_datasetHighPtPhoton30AndZ_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetHighPtPhoton30AndZ_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetHighPtPhoton30AndZ_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetHighPtPhoton30AndZ_selector.triggerConditions = cms.vstring('HLT_HIDoublePhoton15_Eta1p5_Mass50_1000ForPPRef_v8', 
    'HLT_HIDoublePhoton15_Eta1p5_Mass50_1000_R9HECutForPPRef_v8', 
    'HLT_HIDoublePhoton15_Eta2p1_Mass50_1000_R9CutForPPRef_v8', 
    'HLT_HIDoublePhoton15_Eta2p5_Mass50_1000_R9SigmaHECutForPPRef_v8', 
    'HLT_HISinglePhoton30_Eta1p5ForPPRef_v8', 
    'HLT_HISinglePhoton40_Eta1p5ForPPRef_v8', 
    'HLT_HISinglePhoton50_Eta1p5ForPPRef_v8', 
    'HLT_HISinglePhoton60_Eta1p5ForPPRef_v8')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetTOTEM_minBias_selector
streamPhysicsCommissioning_datasetTOTEM_minBias_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetTOTEM_minBias_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetTOTEM_minBias_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetTOTEM_minBias_selector.triggerConditions = cms.vstring('HLT_L1TOTEM1_MinBias_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetTOTEM_zeroBias_selector
streamPhysicsCommissioning_datasetTOTEM_zeroBias_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetTOTEM_zeroBias_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetTOTEM_zeroBias_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetTOTEM_zeroBias_selector.triggerConditions = cms.vstring('HLT_L1TOTEM2_ZeroBias_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetZeroBias_selector
streamPhysicsCommissioning_datasetZeroBias_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetZeroBias_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetZeroBias_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetZeroBias_selector.triggerConditions = cms.vstring('HLT_Random_v3', 
    'HLT_ZeroBias_v6')


# stream PhysicsForward

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsForward_datasetppForward_selector
streamPhysicsForward_datasetppForward_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsForward_datasetppForward_selector.l1tResults = cms.InputTag('')
streamPhysicsForward_datasetppForward_selector.throw      = cms.bool(False)
streamPhysicsForward_datasetppForward_selector.triggerConditions = cms.vstring('HLT_HICastorMediumJetPixel_SingleTrackForPPRef_v5', 
    'HLT_HIL1CastorMediumJetForPPRef_v4', 
    'HLT_HIUPCDoubleMuNotHF2Pixel_SingleTrackForPPRef_v6', 
    'HLT_HIUPCL1DoubleMuOpenNotHF2ForPPRef_v5', 
    'HLT_HIUPCL1MuOpen_NotMinimumBiasHF2_ANDForPPRef_v5', 
    'HLT_HIUPCL1NotMinimumBiasHF2_ANDForPPRef_v5', 
    'HLT_HIUPCL1NotZdcOR_BptxANDForPPRef_v4', 
    'HLT_HIUPCL1ZdcOR_BptxANDForPPRef_v4', 
    'HLT_HIUPCL1ZdcXOR_BptxANDForPPRef_v4', 
    'HLT_HIUPCMuOpen_NotMinimumBiasHF2_ANDPixel_SingleTrackForPPRef_v6', 
    'HLT_HIUPCNotMinimumBiasHF2_ANDPixel_SingleTrackForPPRef_v6', 
    'HLT_HIUPCNotZdcOR_BptxANDPixel_SingleTrackForPPRef_v5', 
    'HLT_HIUPCZdcOR_BptxANDPixel_SingleTrackForPPRef_v5', 
    'HLT_HIUPCZdcXOR_BptxANDPixel_SingleTrackForPPRef_v5')


# stream PhysicsHadronsTaus

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHadronsTaus_datasetBTagCSV_selector
streamPhysicsHadronsTaus_datasetBTagCSV_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHadronsTaus_datasetBTagCSV_selector.l1tResults = cms.InputTag('')
streamPhysicsHadronsTaus_datasetBTagCSV_selector.throw      = cms.bool(False)
streamPhysicsHadronsTaus_datasetBTagCSV_selector.triggerConditions = cms.vstring('HLT_AK4PFBJetBCSV60_Eta2p1ForPPRef_v11', 
    'HLT_AK4PFBJetBCSV80_Eta2p1ForPPRef_v11', 
    'HLT_AK4PFBJetBSSV60_Eta2p1ForPPRef_v11', 
    'HLT_AK4PFBJetBSSV80_Eta2p1ForPPRef_v11')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHadronsTaus_datasetHeavyFlavor_selector
streamPhysicsHadronsTaus_datasetHeavyFlavor_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHadronsTaus_datasetHeavyFlavor_selector.l1tResults = cms.InputTag('')
streamPhysicsHadronsTaus_datasetHeavyFlavor_selector.throw      = cms.bool(False)
streamPhysicsHadronsTaus_datasetHeavyFlavor_selector.triggerConditions = cms.vstring('HLT_DmesonPPTrackingGlobal_Dpt15ForPPRef_v9', 
    'HLT_DmesonPPTrackingGlobal_Dpt20ForPPRef_v9', 
    'HLT_DmesonPPTrackingGlobal_Dpt30ForPPRef_v9', 
    'HLT_DmesonPPTrackingGlobal_Dpt40ForPPRef_v9', 
    'HLT_DmesonPPTrackingGlobal_Dpt50ForPPRef_v9', 
    'HLT_DmesonPPTrackingGlobal_Dpt60ForPPRef_v9', 
    'HLT_DmesonPPTrackingGlobal_Dpt8ForPPRef_v9')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHadronsTaus_datasetHighPtJet80_selector
streamPhysicsHadronsTaus_datasetHighPtJet80_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHadronsTaus_datasetHighPtJet80_selector.l1tResults = cms.InputTag('')
streamPhysicsHadronsTaus_datasetHighPtJet80_selector.throw      = cms.bool(False)
streamPhysicsHadronsTaus_datasetHighPtJet80_selector.triggerConditions = cms.vstring('HLT_AK4CaloJet100_Eta5p1ForPPRef_v8', 
    'HLT_AK4CaloJet100_Jet35_Eta0p7ForPPRef_v8', 
    'HLT_AK4CaloJet100_Jet35_Eta1p1ForPPRef_v8', 
    'HLT_AK4CaloJet110_Eta5p1ForPPRef_v8', 
    'HLT_AK4CaloJet120_Eta5p1ForPPRef_v8', 
    'HLT_AK4CaloJet150ForPPRef_v8', 
    'HLT_AK4CaloJet80_45_45_Eta2p1ForPPRef_v8', 
    'HLT_AK4CaloJet80_Eta5p1ForPPRef_v8', 
    'HLT_AK4CaloJet80_Jet35_Eta0p7ForPPRef_v8', 
    'HLT_AK4CaloJet80_Jet35_Eta1p1ForPPRef_v8', 
    'HLT_AK4PFJet100_Eta5p1ForPPRef_v11', 
    'HLT_AK4PFJet110_Eta5p1ForPPRef_v11', 
    'HLT_AK4PFJet120_Eta5p1ForPPRef_v11', 
    'HLT_AK4PFJet80_Eta5p1ForPPRef_v11')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHadronsTaus_datasetHighPtLowerJets_selector
streamPhysicsHadronsTaus_datasetHighPtLowerJets_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHadronsTaus_datasetHighPtLowerJets_selector.l1tResults = cms.InputTag('')
streamPhysicsHadronsTaus_datasetHighPtLowerJets_selector.throw      = cms.bool(False)
streamPhysicsHadronsTaus_datasetHighPtLowerJets_selector.triggerConditions = cms.vstring('HLT_AK4CaloJet40_Eta5p1ForPPRef_v8', 
    'HLT_AK4CaloJet60_Eta5p1ForPPRef_v8', 
    'HLT_AK4PFJet40_Eta5p1ForPPRef_v11', 
    'HLT_AK4PFJet60_Eta5p1ForPPRef_v11')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsHadronsTaus_datasetJetHT_selector
streamPhysicsHadronsTaus_datasetJetHT_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsHadronsTaus_datasetJetHT_selector.l1tResults = cms.InputTag('')
streamPhysicsHadronsTaus_datasetJetHT_selector.throw      = cms.bool(False)
streamPhysicsHadronsTaus_datasetJetHT_selector.triggerConditions = cms.vstring('HLT_AK4PFDJet60_Eta2p1ForPPRef_v11', 
    'HLT_AK4PFDJet80_Eta2p1ForPPRef_v11')


# stream PhysicsMuons

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsMuons_datasetDoubleMuon_selector
streamPhysicsMuons_datasetDoubleMuon_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsMuons_datasetDoubleMuon_selector.l1tResults = cms.InputTag('')
streamPhysicsMuons_datasetDoubleMuon_selector.throw      = cms.bool(False)
streamPhysicsMuons_datasetDoubleMuon_selector.triggerConditions = cms.vstring('HLT_HIL1DoubleMu0ForPPRef_v4', 
    'HLT_HIL1DoubleMu10ForPPRef_v4', 
    'HLT_HIL2DoubleMu0_NHitQForPPRef_v5', 
    'HLT_HIL3DoubleMu0_OS_m2p5to4p5ForPPRef_v6', 
    'HLT_HIL3DoubleMu0_OS_m7to14ForPPRef_v6')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsMuons_datasetMuPlusX_selector
streamPhysicsMuons_datasetMuPlusX_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsMuons_datasetMuPlusX_selector.l1tResults = cms.InputTag('')
streamPhysicsMuons_datasetMuPlusX_selector.throw      = cms.bool(False)
streamPhysicsMuons_datasetMuPlusX_selector.triggerConditions = cms.vstring('HLT_HIL2Mu3Eta2p5_AK4CaloJet100Eta2p1ForPPRef_v10', 
    'HLT_HIL2Mu3Eta2p5_AK4CaloJet40Eta2p1ForPPRef_v10', 
    'HLT_HIL2Mu3Eta2p5_AK4CaloJet60Eta2p1ForPPRef_v10', 
    'HLT_HIL2Mu3Eta2p5_AK4CaloJet80Eta2p1ForPPRef_v10', 
    'HLT_HIL2Mu3Eta2p5_HIPhoton10Eta1p5ForPPRef_v10', 
    'HLT_HIL2Mu3Eta2p5_HIPhoton15Eta1p5ForPPRef_v10', 
    'HLT_HIL2Mu3Eta2p5_HIPhoton20Eta1p5ForPPRef_v10', 
    'HLT_HIL2Mu3Eta2p5_HIPhoton30Eta1p5ForPPRef_v10', 
    'HLT_HIL2Mu3Eta2p5_HIPhoton40Eta1p5ForPPRef_v10')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsMuons_datasetSingleMuHighPt_selector
streamPhysicsMuons_datasetSingleMuHighPt_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsMuons_datasetSingleMuHighPt_selector.l1tResults = cms.InputTag('')
streamPhysicsMuons_datasetSingleMuHighPt_selector.throw      = cms.bool(False)
streamPhysicsMuons_datasetSingleMuHighPt_selector.triggerConditions = cms.vstring('HLT_HIL2Mu15ForPPRef_v6', 
    'HLT_HIL2Mu20ForPPRef_v6', 
    'HLT_HIL2Mu5_NHitQ10ForPPRef_v6', 
    'HLT_HIL2Mu7_NHitQ10ForPPRef_v6', 
    'HLT_HIL3Mu15ForPPRef_v6', 
    'HLT_HIL3Mu20ForPPRef_v6', 
    'HLT_HIL3Mu5_NHitQ15ForPPRef_v6', 
    'HLT_HIL3Mu7_NHitQ15ForPPRef_v6')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsMuons_datasetSingleMuLowPt_selector
streamPhysicsMuons_datasetSingleMuLowPt_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsMuons_datasetSingleMuLowPt_selector.l1tResults = cms.InputTag('')
streamPhysicsMuons_datasetSingleMuLowPt_selector.throw      = cms.bool(False)
streamPhysicsMuons_datasetSingleMuLowPt_selector.triggerConditions = cms.vstring('HLT_HIL2Mu3_NHitQ10ForPPRef_v6', 
    'HLT_HIL3Mu3_NHitQ15ForPPRef_v6')


# stream PhysicsTracks

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsTracks_datasetFullTrack_selector
streamPhysicsTracks_datasetFullTrack_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsTracks_datasetFullTrack_selector.l1tResults = cms.InputTag('')
streamPhysicsTracks_datasetFullTrack_selector.throw      = cms.bool(False)
streamPhysicsTracks_datasetFullTrack_selector.triggerConditions = cms.vstring('HLT_FullTrack18ForPPRef_v9', 
    'HLT_FullTrack24ForPPRef_v9', 
    'HLT_FullTrack34ForPPRef_v9', 
    'HLT_FullTrack45ForPPRef_v9', 
    'HLT_FullTrack53ForPPRef_v9')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsTracks_datasetHighMultiplicity_selector
streamPhysicsTracks_datasetHighMultiplicity_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsTracks_datasetHighMultiplicity_selector.l1tResults = cms.InputTag('')
streamPhysicsTracks_datasetHighMultiplicity_selector.throw      = cms.bool(False)
streamPhysicsTracks_datasetHighMultiplicity_selector.triggerConditions = cms.vstring('HLT_PixelTracks_Multiplicity110ForPPRef_v5', 
    'HLT_PixelTracks_Multiplicity135ForPPRef_v5', 
    'HLT_PixelTracks_Multiplicity160ForPPRef_v5', 
    'HLT_PixelTracks_Multiplicity60ForPPRef_v5', 
    'HLT_PixelTracks_Multiplicity85ForPPRef_v5')

