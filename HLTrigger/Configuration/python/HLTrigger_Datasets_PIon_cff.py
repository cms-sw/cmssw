# /dev/CMSSW_8_0_0/PIon

import FWCore.ParameterSet.Config as cms


# stream PhysicsCommissioning

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetHLTPhysics_selector
streamPhysicsCommissioning_datasetHLTPhysics_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetHLTPhysics_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetHLTPhysics_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetHLTPhysics_selector.triggerConditions = cms.vstring('HLT_L1FatEvents_v2', 
    'HLT_Physics_v5')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetHcalNZS_selector
streamPhysicsCommissioning_datasetHcalNZS_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetHcalNZS_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetHcalNZS_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetHcalNZS_selector.triggerConditions = cms.vstring('HLT_PAHcalNZS_v1', 
    'HLT_PAHcalPhiSym_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetZeroBias_selector
streamPhysicsCommissioning_datasetZeroBias_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetZeroBias_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetZeroBias_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetZeroBias_selector.triggerConditions = cms.vstring('HLT_PARandom_v1', 
    'HLT_PAZeroBias_IsolatedBunches_v1', 
    'HLT_PAZeroBias_v1')


# stream PhysicsPACastor

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPACastor_datasetPACastor_selector
streamPhysicsPACastor_datasetPACastor_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPACastor_datasetPACastor_selector.l1tResults = cms.InputTag('')
streamPhysicsPACastor_datasetPACastor_selector.throw      = cms.bool(False)
streamPhysicsPACastor_datasetPACastor_selector.triggerConditions = cms.vstring('HLT_PAL1CastorHaloMuon_v1', 
    'HLT_PAL1CastorMediumJet_BptxAND_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPACastor_datasetPAForward_selector
streamPhysicsPACastor_datasetPAForward_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPACastor_datasetPAForward_selector.l1tResults = cms.InputTag('')
streamPhysicsPACastor_datasetPAForward_selector.throw      = cms.bool(False)
streamPhysicsPACastor_datasetPAForward_selector.triggerConditions = cms.vstring('HLT_PADoubleEG2_HFOneTowerVeto_SingleTrack_v1', 
    'HLT_PADoubleEG2_HFOneTowerVeto_v1', 
    'HLT_PADoubleEG2_HFTwoTowerVeto_SingleTrack_v1', 
    'HLT_PADoubleEG2_HFTwoTowerVeto_v1', 
    'HLT_PADoubleMuOpen_HFOneTowerVeto_SingleTrack_v1', 
    'HLT_PADoubleMuOpen_HFOneTowerVeto_v1', 
    'HLT_PADoubleMuOpen_HFThreeTowerVeto_SingleTrack_v1', 
    'HLT_PADoubleMuOpen_HFThreeTowerVeto_v1', 
    'HLT_PADoubleMuOpen_HFTwoTowerVeto_SingleTrack_v1', 
    'HLT_PADoubleMuOpen_HFTwoTowerVeto_v1', 
    'HLT_PASingleEG5_HFOneTowerVeto_SingleTrack_v1', 
    'HLT_PASingleEG5_HFOneTowerVeto_v1', 
    'HLT_PASingleEG5_HFThreeTowerVeto_SingleTrack_v1', 
    'HLT_PASingleEG5_HFThreeTowerVeto_v1', 
    'HLT_PASingleEG5_HFTwoTowerVeto_SingleTrack_v1', 
    'HLT_PASingleEG5_HFTwoTowerVeto_v1', 
    'HLT_PASingleMuOpen_HFOneTowerVeto_SingleTrack_v1', 
    'HLT_PASingleMuOpen_HFOneTowerVeto_v1', 
    'HLT_PASingleMuOpen_HFThreeTowerVeto_SingleTrack_v1', 
    'HLT_PASingleMuOpen_HFThreeTowerVeto_v1', 
    'HLT_PASingleMuOpen_HFTwoTowerVeto_SingleTrack_v1', 
    'HLT_PASingleMuOpen_HFTwoTowerVeto_v1', 
    'HLT_PASingleMuOpen_PixelTrackGt0Lt10_v1', 
    'HLT_PASingleMuOpen_PixelTrackGt0Lt15_v1', 
    'HLT_PASingleMuOpen_PixelTrackGt0_FullTrackLt10_v1', 
    'HLT_PASingleMuOpen_PixelTrackGt0_FullTrackLt15_v1', 
    'HLT_PASingleMuOpen_v1')


# stream PhysicsPAHighMultiplicity0

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPAHighMultiplicity0_datasetPAHighMultiplicity0_selector
streamPhysicsPAHighMultiplicity0_datasetPAHighMultiplicity0_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPAHighMultiplicity0_datasetPAHighMultiplicity0_selector.l1tResults = cms.InputTag('')
streamPhysicsPAHighMultiplicity0_datasetPAHighMultiplicity0_selector.throw      = cms.bool(False)
streamPhysicsPAHighMultiplicity0_datasetPAHighMultiplicity0_selector.triggerConditions = cms.vstring('HLT_PAFullTracks_HFSumEt005_HighPt16_v3', 
    'HLT_PAFullTracks_HFSumEt005_HighPt8_v3', 
    'HLT_PAFullTracks_Multiplicity110_HighPt16_v3', 
    'HLT_PAFullTracks_Multiplicity110_HighPt8_v2', 
    'HLT_PAFullTracks_Multiplicity120_v1', 
    'HLT_PAFullTracks_Multiplicity150_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPAHighMultiplicity0_datasetPAHighMultiplicity7_selector
streamPhysicsPAHighMultiplicity0_datasetPAHighMultiplicity7_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPAHighMultiplicity0_datasetPAHighMultiplicity7_selector.l1tResults = cms.InputTag('')
streamPhysicsPAHighMultiplicity0_datasetPAHighMultiplicity7_selector.throw      = cms.bool(False)
streamPhysicsPAHighMultiplicity0_datasetPAHighMultiplicity7_selector.triggerConditions = cms.vstring('HLT_PAFullTracks_Multiplicity220_v5', 
    'HLT_PAFullTracks_Multiplicity250_v5', 
    'HLT_PAFullTracks_Multiplicity280_v5')


# stream PhysicsPAHighMultiplicity1

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPAHighMultiplicity1_datasetPAHighMultiplicity1_selector
streamPhysicsPAHighMultiplicity1_datasetPAHighMultiplicity1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPAHighMultiplicity1_datasetPAHighMultiplicity1_selector.l1tResults = cms.InputTag('')
streamPhysicsPAHighMultiplicity1_datasetPAHighMultiplicity1_selector.throw      = cms.bool(False)
streamPhysicsPAHighMultiplicity1_datasetPAHighMultiplicity1_selector.triggerConditions = cms.vstring('HLT_PAFullTracks_Multiplicity185_part1_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPAHighMultiplicity1_datasetPAHighMultiplicity2_selector
streamPhysicsPAHighMultiplicity1_datasetPAHighMultiplicity2_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPAHighMultiplicity1_datasetPAHighMultiplicity2_selector.l1tResults = cms.InputTag('')
streamPhysicsPAHighMultiplicity1_datasetPAHighMultiplicity2_selector.throw      = cms.bool(False)
streamPhysicsPAHighMultiplicity1_datasetPAHighMultiplicity2_selector.triggerConditions = cms.vstring('HLT_PAFullTracks_Multiplicity185_part2_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPAHighMultiplicity1_datasetPAHighMultiplicity3_selector
streamPhysicsPAHighMultiplicity1_datasetPAHighMultiplicity3_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPAHighMultiplicity1_datasetPAHighMultiplicity3_selector.l1tResults = cms.InputTag('')
streamPhysicsPAHighMultiplicity1_datasetPAHighMultiplicity3_selector.throw      = cms.bool(False)
streamPhysicsPAHighMultiplicity1_datasetPAHighMultiplicity3_selector.triggerConditions = cms.vstring('HLT_PAFullTracks_Multiplicity185_part3_v4')


# stream PhysicsPAHighMultiplicity2

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPAHighMultiplicity2_datasetPAHighMultiplicity4_selector
streamPhysicsPAHighMultiplicity2_datasetPAHighMultiplicity4_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPAHighMultiplicity2_datasetPAHighMultiplicity4_selector.l1tResults = cms.InputTag('')
streamPhysicsPAHighMultiplicity2_datasetPAHighMultiplicity4_selector.throw      = cms.bool(False)
streamPhysicsPAHighMultiplicity2_datasetPAHighMultiplicity4_selector.triggerConditions = cms.vstring('HLT_PAFullTracks_Multiplicity185_part4_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPAHighMultiplicity2_datasetPAHighMultiplicity5_selector
streamPhysicsPAHighMultiplicity2_datasetPAHighMultiplicity5_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPAHighMultiplicity2_datasetPAHighMultiplicity5_selector.l1tResults = cms.InputTag('')
streamPhysicsPAHighMultiplicity2_datasetPAHighMultiplicity5_selector.throw      = cms.bool(False)
streamPhysicsPAHighMultiplicity2_datasetPAHighMultiplicity5_selector.triggerConditions = cms.vstring('HLT_PAFullTracks_Multiplicity185_part5_v4')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPAHighMultiplicity2_datasetPAHighMultiplicity6_selector
streamPhysicsPAHighMultiplicity2_datasetPAHighMultiplicity6_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPAHighMultiplicity2_datasetPAHighMultiplicity6_selector.l1tResults = cms.InputTag('')
streamPhysicsPAHighMultiplicity2_datasetPAHighMultiplicity6_selector.throw      = cms.bool(False)
streamPhysicsPAHighMultiplicity2_datasetPAHighMultiplicity6_selector.triggerConditions = cms.vstring('HLT_PAFullTracks_Multiplicity185_part6_v4')


# stream PhysicsPAHighPt1

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPAHighPt1_datasetPADTrack1_selector
streamPhysicsPAHighPt1_datasetPADTrack1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPAHighPt1_datasetPADTrack1_selector.l1tResults = cms.InputTag('')
streamPhysicsPAHighPt1_datasetPADTrack1_selector.throw      = cms.bool(False)
streamPhysicsPAHighPt1_datasetPADTrack1_selector.triggerConditions = cms.vstring('HLT_PADmesonPPTrackingGlobal_Dpt15_v3', 
    'HLT_PADmesonPPTrackingGlobal_Dpt30_v2', 
    'HLT_PADmesonPPTrackingGlobal_Dpt50_v2', 
    'HLT_PADmesonPPTrackingGlobal_Dpt55_v1', 
    'HLT_PADmesonPPTrackingGlobal_Dpt5_v2', 
    'HLT_PADmesonPPTrackingGlobal_Dpt8_v2', 
    'HLT_PAFullTracks_HighPt20_v3', 
    'HLT_PAFullTracks_HighPt30_v1', 
    'HLT_PAFullTracks_HighPt40_v1', 
    'HLT_PAFullTracks_HighPt50_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPAHighPt1_datasetPAEGJet1_selector
streamPhysicsPAHighPt1_datasetPAEGJet1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPAHighPt1_datasetPAEGJet1_selector.l1tResults = cms.InputTag('')
streamPhysicsPAHighPt1_datasetPAEGJet1_selector.throw      = cms.bool(False)
streamPhysicsPAHighPt1_datasetPAEGJet1_selector.triggerConditions = cms.vstring('HLT_PAAK4CaloBJetCSV40_Eta2p1_v3', 
    'HLT_PAAK4CaloBJetCSV60_Eta2p1_v3', 
    'HLT_PAAK4CaloBJetCSV80_Eta2p1_v3', 
    'HLT_PAAK4CaloJet100_Eta5p1_v3', 
    'HLT_PAAK4CaloJet40_Eta1p9toEta5p1_v3', 
    'HLT_PAAK4CaloJet40_Eta2p9toEta5p1_v3', 
    'HLT_PAAK4CaloJet40_Eta5p1_SeededWithMB_v3', 
    'HLT_PAAK4CaloJet40_Eta5p1_v3', 
    'HLT_PAAK4CaloJet60_Eta1p9toEta5p1_v3', 
    'HLT_PAAK4CaloJet60_Eta5p1_v3', 
    'HLT_PAAK4CaloJet80_Eta5p1_v3', 
    'HLT_PAAK4PFBJetCSV40_CommonTracking_Eta2p1_v3', 
    'HLT_PAAK4PFBJetCSV40_Eta2p1_v3', 
    'HLT_PAAK4PFBJetCSV60_CommonTracking_Eta2p1_v3', 
    'HLT_PAAK4PFBJetCSV60_Eta2p1_v3', 
    'HLT_PAAK4PFBJetCSV80_CommonTracking_Eta2p1_v3', 
    'HLT_PAAK4PFBJetCSV80_Eta2p1_v3', 
    'HLT_PAAK4PFJet100_Eta5p1_v3', 
    'HLT_PAAK4PFJet120_Eta5p1_v2', 
    'HLT_PAAK4PFJet40_Eta1p9toEta5p1_v3', 
    'HLT_PAAK4PFJet40_Eta2p9toEta5p1_v3', 
    'HLT_PAAK4PFJet40_Eta5p1_SeededWithMB_v3', 
    'HLT_PAAK4PFJet40_Eta5p1_v3', 
    'HLT_PAAK4PFJet60_Eta1p9toEta5p1_v3', 
    'HLT_PAAK4PFJet60_Eta5p1_v4', 
    'HLT_PAAK4PFJet80_Eta5p1_v3', 
    'HLT_PADiAK4CaloJetAve40_Eta5p1_v3', 
    'HLT_PADiAK4CaloJetAve60_Eta5p1_v3', 
    'HLT_PADiAK4CaloJetAve80_Eta5p1_v3', 
    'HLT_PADiAK4PFJetAve40_Eta5p1_v3', 
    'HLT_PADiAK4PFJetAve60_Eta5p1_v3', 
    'HLT_PADiAK4PFJetAve80_Eta5p1_v3', 
    'HLT_PADoublePhoton15_Eta3p1_Mass50_1000_v2', 
    'HLT_PAEle20_WPLoose_Gsf_v1', 
    'HLT_PAIsoPhoton20_Eta3p1_PPStyle_v2', 
    'HLT_PAPhoton10_Eta3p1_PPStyle_v1', 
    'HLT_PAPhoton15_Eta3p1_PPStyle_v1', 
    'HLT_PAPhoton20_Eta3p1_PPStyle_v1', 
    'HLT_PAPhoton30_Eta3p1_PPStyle_v1', 
    'HLT_PAPhoton40_Eta3p1_PPStyle_v1', 
    'HLT_PASingleIsoPhoton20_Eta3p1_v2', 
    'HLT_PASinglePhoton10_Eta3p1_v1', 
    'HLT_PASinglePhoton15_Eta3p1_SeededWithMB_v1', 
    'HLT_PASinglePhoton15_Eta3p1_v1', 
    'HLT_PASinglePhoton20_Eta3p1_SeededWithMB_v1', 
    'HLT_PASinglePhoton20_Eta3p1_v1', 
    'HLT_PASinglePhoton30_Eta3p1_v1', 
    'HLT_PASinglePhoton30_L1EGJet_Eta3p1_v1', 
    'HLT_PASinglePhoton40_Eta3p1_v1', 
    'HLT_PASinglePhoton40_L1EGJet_Eta3p1_v1')


# stream PhysicsPAHighPt2

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPAHighPt2_datasetPADTrack2_selector
streamPhysicsPAHighPt2_datasetPADTrack2_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPAHighPt2_datasetPADTrack2_selector.l1tResults = cms.InputTag('')
streamPhysicsPAHighPt2_datasetPADTrack2_selector.throw      = cms.bool(False)
streamPhysicsPAHighPt2_datasetPADTrack2_selector.triggerConditions = cms.vstring('HLT_PADmesonPPTrackingGlobal_Dpt5_part2_v2', 
    'HLT_PADmesonPPTrackingGlobal_Dpt5_part3_v2')


# stream PhysicsPAMinimumBias0

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPAMinimumBias0_datasetPAEmptyBX_selector
streamPhysicsPAMinimumBias0_datasetPAEmptyBX_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPAMinimumBias0_datasetPAEmptyBX_selector.l1tResults = cms.InputTag('')
streamPhysicsPAMinimumBias0_datasetPAEmptyBX_selector.throw      = cms.bool(False)
streamPhysicsPAMinimumBias0_datasetPAEmptyBX_selector.triggerConditions = cms.vstring('HLT_PABptxXOR_v1', 
    'HLT_PAL1BptxMinusNotBptxPlus_v1', 
    'HLT_PAL1BptxPlusNotBptxMinus_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPAMinimumBias0_datasetPAMinimumBias1_selector
streamPhysicsPAMinimumBias0_datasetPAMinimumBias1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPAMinimumBias0_datasetPAMinimumBias1_selector.l1tResults = cms.InputTag('')
streamPhysicsPAMinimumBias0_datasetPAMinimumBias1_selector.throw      = cms.bool(False)
streamPhysicsPAMinimumBias0_datasetPAMinimumBias1_selector.triggerConditions = cms.vstring('HLT_PAL1MinimumBiasHF_OR_SinglePixelTrack_part1_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPAMinimumBias0_datasetPAMinimumBias2_selector
streamPhysicsPAMinimumBias0_datasetPAMinimumBias2_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPAMinimumBias0_datasetPAMinimumBias2_selector.l1tResults = cms.InputTag('')
streamPhysicsPAMinimumBias0_datasetPAMinimumBias2_selector.throw      = cms.bool(False)
streamPhysicsPAMinimumBias0_datasetPAMinimumBias2_selector.triggerConditions = cms.vstring('HLT_PAL1MinimumBiasHF_OR_SinglePixelTrack_part2_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPAMinimumBias0_datasetPAMinimumBiasBkg_selector
streamPhysicsPAMinimumBias0_datasetPAMinimumBiasBkg_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPAMinimumBias0_datasetPAMinimumBiasBkg_selector.l1tResults = cms.InputTag('')
streamPhysicsPAMinimumBias0_datasetPAMinimumBiasBkg_selector.throw      = cms.bool(False)
streamPhysicsPAMinimumBias0_datasetPAMinimumBiasBkg_selector.triggerConditions = cms.vstring('HLT_PAL1BptxMinus_v1', 
    'HLT_PAL1BptxPlus_v1', 
    'HLT_PAL1MinimumBiasHF_AND_SinglePixelTrack_v1', 
    'HLT_PAL1MinimumBiasHF_AND_v1', 
    'HLT_PAL1MinimumBiasHF_OR_SinglePixelTrack_ForSkim_v1', 
    'HLT_PAL1MinimumBiasHF_OR_SinglePixelTrack_v1', 
    'HLT_PAL1MinimumBiasHF_OR_v1', 
    'HLT_PAZeroBias_DoublePixelTrack_v1', 
    'HLT_PAZeroBias_SinglePixelTrack_v1')


# stream PhysicsPAMinimumBias1

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPAMinimumBias1_datasetPAMinimumBias3_selector
streamPhysicsPAMinimumBias1_datasetPAMinimumBias3_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPAMinimumBias1_datasetPAMinimumBias3_selector.l1tResults = cms.InputTag('')
streamPhysicsPAMinimumBias1_datasetPAMinimumBias3_selector.throw      = cms.bool(False)
streamPhysicsPAMinimumBias1_datasetPAMinimumBias3_selector.triggerConditions = cms.vstring('HLT_PAL1MinimumBiasHF_OR_SinglePixelTrack_part3_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPAMinimumBias1_datasetPAMinimumBias4_selector
streamPhysicsPAMinimumBias1_datasetPAMinimumBias4_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPAMinimumBias1_datasetPAMinimumBias4_selector.l1tResults = cms.InputTag('')
streamPhysicsPAMinimumBias1_datasetPAMinimumBias4_selector.throw      = cms.bool(False)
streamPhysicsPAMinimumBias1_datasetPAMinimumBias4_selector.triggerConditions = cms.vstring('HLT_PAL1MinimumBiasHF_OR_SinglePixelTrack_part4_v2')


# stream PhysicsPAMinimumBias2

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPAMinimumBias2_datasetPAMinimumBias5_selector
streamPhysicsPAMinimumBias2_datasetPAMinimumBias5_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPAMinimumBias2_datasetPAMinimumBias5_selector.l1tResults = cms.InputTag('')
streamPhysicsPAMinimumBias2_datasetPAMinimumBias5_selector.throw      = cms.bool(False)
streamPhysicsPAMinimumBias2_datasetPAMinimumBias5_selector.triggerConditions = cms.vstring('HLT_PAL1MinimumBiasHF_OR_SinglePixelTrack_part5_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPAMinimumBias2_datasetPAMinimumBias6_selector
streamPhysicsPAMinimumBias2_datasetPAMinimumBias6_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPAMinimumBias2_datasetPAMinimumBias6_selector.l1tResults = cms.InputTag('')
streamPhysicsPAMinimumBias2_datasetPAMinimumBias6_selector.throw      = cms.bool(False)
streamPhysicsPAMinimumBias2_datasetPAMinimumBias6_selector.triggerConditions = cms.vstring('HLT_PAL1MinimumBiasHF_OR_SinglePixelTrack_part6_v2')


# stream PhysicsPAMinimumBias3

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPAMinimumBias3_datasetPAMinimumBias7_selector
streamPhysicsPAMinimumBias3_datasetPAMinimumBias7_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPAMinimumBias3_datasetPAMinimumBias7_selector.l1tResults = cms.InputTag('')
streamPhysicsPAMinimumBias3_datasetPAMinimumBias7_selector.throw      = cms.bool(False)
streamPhysicsPAMinimumBias3_datasetPAMinimumBias7_selector.triggerConditions = cms.vstring('HLT_PAL1MinimumBiasHF_OR_SinglePixelTrack_part7_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPAMinimumBias3_datasetPAMinimumBias8_selector
streamPhysicsPAMinimumBias3_datasetPAMinimumBias8_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPAMinimumBias3_datasetPAMinimumBias8_selector.l1tResults = cms.InputTag('')
streamPhysicsPAMinimumBias3_datasetPAMinimumBias8_selector.throw      = cms.bool(False)
streamPhysicsPAMinimumBias3_datasetPAMinimumBias8_selector.triggerConditions = cms.vstring('HLT_PAL1MinimumBiasHF_OR_SinglePixelTrack_part8_v2')


# stream PhysicsPAMuons

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPAMuons_datasetPADoubleMuon_selector
streamPhysicsPAMuons_datasetPADoubleMuon_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPAMuons_datasetPADoubleMuon_selector.l1tResults = cms.InputTag('')
streamPhysicsPAMuons_datasetPADoubleMuon_selector.throw      = cms.bool(False)
streamPhysicsPAMuons_datasetPADoubleMuon_selector.triggerConditions = cms.vstring('HLT_PAL1DoubleMu0_HighQ_v1', 
    'HLT_PAL1DoubleMu0_MGT1_v1', 
    'HLT_PAL1DoubleMu0_v1', 
    'HLT_PAL1DoubleMu10_v1', 
    'HLT_PAL1DoubleMuOpen_OS_v1', 
    'HLT_PAL1DoubleMuOpen_SS_v1', 
    'HLT_PAL1DoubleMuOpen_v1', 
    'HLT_PAL2DoubleMu0_v1', 
    'HLT_PAL2DoubleMu10_v1', 
    'HLT_PAL3DoubleMu0_HIon_v1', 
    'HLT_PAL3DoubleMu0_v1', 
    'HLT_PAL3DoubleMu10_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPAMuons_datasetPASingleMuon_selector
streamPhysicsPAMuons_datasetPASingleMuon_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPAMuons_datasetPASingleMuon_selector.l1tResults = cms.InputTag('')
streamPhysicsPAMuons_datasetPASingleMuon_selector.throw      = cms.bool(False)
streamPhysicsPAMuons_datasetPASingleMuon_selector.triggerConditions = cms.vstring('HLT_PAAK4CaloJet30_Eta5p1_PAL3Mu3_v4', 
    'HLT_PAAK4CaloJet30_Eta5p1_PAL3Mu5_v3', 
    'HLT_PAAK4CaloJet40_Eta5p1_PAL3Mu3_v4', 
    'HLT_PAAK4CaloJet40_Eta5p1_PAL3Mu5_v3', 
    'HLT_PAAK4CaloJet60_Eta5p1_PAL3Mu3_v4', 
    'HLT_PAAK4CaloJet60_Eta5p1_PAL3Mu5_v3', 
    'HLT_PAL2Mu12_v1', 
    'HLT_PAL2Mu15_v1', 
    'HLT_PAL3Mu12_v1', 
    'HLT_PAL3Mu15_v1', 
    'HLT_PAL3Mu3_v1', 
    'HLT_PAL3Mu5_v3', 
    'HLT_PAL3Mu7_v1', 
    'HLT_PASinglePhoton10_Eta3p1_PAL3Mu3_v2', 
    'HLT_PASinglePhoton10_Eta3p1_PAL3Mu5_v2', 
    'HLT_PASinglePhoton15_Eta3p1_PAL3Mu3_v2', 
    'HLT_PASinglePhoton15_Eta3p1_PAL3Mu5_v2', 
    'HLT_PASinglePhoton20_Eta3p1_PAL3Mu3_v2', 
    'HLT_PASinglePhoton20_Eta3p1_PAL3Mu5_v2')

