# /dev/CMSSW_8_0_0/PIon

import FWCore.ParameterSet.Config as cms


# stream PhysicsCommissioning

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsCommissioning_datasetHLTPhysics_selector
streamPhysicsCommissioning_datasetHLTPhysics_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsCommissioning_datasetHLTPhysics_selector.l1tResults = cms.InputTag('')
streamPhysicsCommissioning_datasetHLTPhysics_selector.throw      = cms.bool(False)
streamPhysicsCommissioning_datasetHLTPhysics_selector.triggerConditions = cms.vstring('HLT_Physics_v5')


# stream PhysicsPAHighMultiplicity0

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPAHighMultiplicity0_datasetPAHighMultiplicity0_selector
streamPhysicsPAHighMultiplicity0_datasetPAHighMultiplicity0_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPAHighMultiplicity0_datasetPAHighMultiplicity0_selector.l1tResults = cms.InputTag('')
streamPhysicsPAHighMultiplicity0_datasetPAHighMultiplicity0_selector.throw      = cms.bool(False)
streamPhysicsPAHighMultiplicity0_datasetPAHighMultiplicity0_selector.triggerConditions = cms.vstring('HLT_PAFullTracks_Multiplicity110_HighPt16_v1', 
    'HLT_PAFullTracks_Multiplicity110_HighPt8_v1', 
    'HLT_PAFullTracks_Multiplicity120_v1', 
    'HLT_PAFullTracks_Multiplicity150_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPAHighMultiplicity0_datasetPAHighMultiplicity7_selector
streamPhysicsPAHighMultiplicity0_datasetPAHighMultiplicity7_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPAHighMultiplicity0_datasetPAHighMultiplicity7_selector.l1tResults = cms.InputTag('')
streamPhysicsPAHighMultiplicity0_datasetPAHighMultiplicity7_selector.throw      = cms.bool(False)
streamPhysicsPAHighMultiplicity0_datasetPAHighMultiplicity7_selector.triggerConditions = cms.vstring('HLT_PAFullTracks_Multiplicity220_v1', 
    'HLT_PAFullTracks_Multiplicity250_v1', 
    'HLT_PAFullTracks_Multiplicity280_v1')


# stream PhysicsPAHighMultiplicity1

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPAHighMultiplicity1_datasetPAHighMultiplicity1_selector
streamPhysicsPAHighMultiplicity1_datasetPAHighMultiplicity1_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPAHighMultiplicity1_datasetPAHighMultiplicity1_selector.l1tResults = cms.InputTag('')
streamPhysicsPAHighMultiplicity1_datasetPAHighMultiplicity1_selector.throw      = cms.bool(False)
streamPhysicsPAHighMultiplicity1_datasetPAHighMultiplicity1_selector.triggerConditions = cms.vstring('HLT_PAFullTracks_Multiplicity185_part1_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPAHighMultiplicity1_datasetPAHighMultiplicity2_selector
streamPhysicsPAHighMultiplicity1_datasetPAHighMultiplicity2_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPAHighMultiplicity1_datasetPAHighMultiplicity2_selector.l1tResults = cms.InputTag('')
streamPhysicsPAHighMultiplicity1_datasetPAHighMultiplicity2_selector.throw      = cms.bool(False)
streamPhysicsPAHighMultiplicity1_datasetPAHighMultiplicity2_selector.triggerConditions = cms.vstring('HLT_PAFullTracks_Multiplicity185_part2_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPAHighMultiplicity1_datasetPAHighMultiplicity3_selector
streamPhysicsPAHighMultiplicity1_datasetPAHighMultiplicity3_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPAHighMultiplicity1_datasetPAHighMultiplicity3_selector.l1tResults = cms.InputTag('')
streamPhysicsPAHighMultiplicity1_datasetPAHighMultiplicity3_selector.throw      = cms.bool(False)
streamPhysicsPAHighMultiplicity1_datasetPAHighMultiplicity3_selector.triggerConditions = cms.vstring('HLT_PAFullTracks_Multiplicity185_part3_v1')


# stream PhysicsPAHighMultiplicity2

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPAHighMultiplicity2_datasetPAHighMultiplicity4_selector
streamPhysicsPAHighMultiplicity2_datasetPAHighMultiplicity4_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPAHighMultiplicity2_datasetPAHighMultiplicity4_selector.l1tResults = cms.InputTag('')
streamPhysicsPAHighMultiplicity2_datasetPAHighMultiplicity4_selector.throw      = cms.bool(False)
streamPhysicsPAHighMultiplicity2_datasetPAHighMultiplicity4_selector.triggerConditions = cms.vstring('HLT_PAFullTracks_Multiplicity185_part4_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPAHighMultiplicity2_datasetPAHighMultiplicity5_selector
streamPhysicsPAHighMultiplicity2_datasetPAHighMultiplicity5_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPAHighMultiplicity2_datasetPAHighMultiplicity5_selector.l1tResults = cms.InputTag('')
streamPhysicsPAHighMultiplicity2_datasetPAHighMultiplicity5_selector.throw      = cms.bool(False)
streamPhysicsPAHighMultiplicity2_datasetPAHighMultiplicity5_selector.triggerConditions = cms.vstring('HLT_PAFullTracks_Multiplicity185_part5_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPAHighMultiplicity2_datasetPAHighMultiplicity6_selector
streamPhysicsPAHighMultiplicity2_datasetPAHighMultiplicity6_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPAHighMultiplicity2_datasetPAHighMultiplicity6_selector.l1tResults = cms.InputTag('')
streamPhysicsPAHighMultiplicity2_datasetPAHighMultiplicity6_selector.throw      = cms.bool(False)
streamPhysicsPAHighMultiplicity2_datasetPAHighMultiplicity6_selector.triggerConditions = cms.vstring('HLT_PAFullTracks_Multiplicity185_part6_v1')


# stream PhysicsPAJetsEG

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPAJetsEG_datasetPAHigherEG_selector
streamPhysicsPAJetsEG_datasetPAHigherEG_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPAJetsEG_datasetPAHigherEG_selector.l1tResults = cms.InputTag('')
streamPhysicsPAJetsEG_datasetPAHigherEG_selector.throw      = cms.bool(False)
streamPhysicsPAJetsEG_datasetPAHigherEG_selector.triggerConditions = cms.vstring('HLT_PADoublePhoton15_Eta3p1_Mass50_1000_v2', 
    'HLT_PASinglePhoton30_Eta3p1_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPAJetsEG_datasetPAHigherJets_selector
streamPhysicsPAJetsEG_datasetPAHigherJets_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPAJetsEG_datasetPAHigherJets_selector.l1tResults = cms.InputTag('')
streamPhysicsPAJetsEG_datasetPAHigherJets_selector.throw      = cms.bool(False)
streamPhysicsPAJetsEG_datasetPAHigherJets_selector.triggerConditions = cms.vstring('HLT_PAAK4CaloJet60_Eta1p9toEta5p1_v2', 
    'HLT_PAAK4CaloJet80_Eta5p1_v2', 
    'HLT_PAAK4PFJet60_Eta1p9toEta5p1_v2', 
    'HLT_PAAK4PFJet80_Eta5p1_v2', 
    'HLT_PADiAK4CaloJetAve60_Eta5p1_v2', 
    'HLT_PADiAK4CaloJetAve80_Eta5p1_v2', 
    'HLT_PADiAK4PFJetAve60_Eta5p1_v2', 
    'HLT_PADiAK4PFJetAve80_Eta5p1_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPAJetsEG_datasetPALowerEG_selector
streamPhysicsPAJetsEG_datasetPALowerEG_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPAJetsEG_datasetPALowerEG_selector.l1tResults = cms.InputTag('')
streamPhysicsPAJetsEG_datasetPALowerEG_selector.throw      = cms.bool(False)
streamPhysicsPAJetsEG_datasetPALowerEG_selector.triggerConditions = cms.vstring('HLT_PASinglePhoton10_Eta3p1_v2', 
    'HLT_PASinglePhoton15_Eta3p1_v2', 
    'HLT_PASinglePhoton20_Eta3p1_v2')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPAJetsEG_datasetPALowerJets_selector
streamPhysicsPAJetsEG_datasetPALowerJets_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPAJetsEG_datasetPALowerJets_selector.l1tResults = cms.InputTag('')
streamPhysicsPAJetsEG_datasetPALowerJets_selector.throw      = cms.bool(False)
streamPhysicsPAJetsEG_datasetPALowerJets_selector.triggerConditions = cms.vstring('HLT_PAAK4CaloJet40_Eta1p9toEta5p1_v2', 
    'HLT_PAAK4CaloJet40_Eta2p9toEta5p1_v2', 
    'HLT_PAAK4CaloJet40_Eta5p1_v2', 
    'HLT_PAAK4CaloJet60_Eta5p1_v2', 
    'HLT_PAAK4PFJet40_Eta1p9toEta5p1_v2', 
    'HLT_PAAK4PFJet40_Eta2p9toEta5p1_v2', 
    'HLT_PAAK4PFJet40_Eta5p1_v2', 
    'HLT_PAAK4PFJet60_Eta5p1_v2', 
    'HLT_PADiAK4CaloJetAve40_Eta5p1_v2', 
    'HLT_PADiAK4PFJetAve40_Eta5p1_v2')


# stream PhysicsPAMesonD

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPAMesonD_datasetPAMesonD_selector
streamPhysicsPAMesonD_datasetPAMesonD_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPAMesonD_datasetPAMesonD_selector.l1tResults = cms.InputTag('')
streamPhysicsPAMesonD_datasetPAMesonD_selector.throw      = cms.bool(False)
streamPhysicsPAMesonD_datasetPAMesonD_selector.triggerConditions = cms.vstring('HLT_PADmesonPPTrackingGlobal_Dpt15_v1', 
    'HLT_PADmesonPPTrackingGlobal_Dpt30_v1', 
    'HLT_PADmesonPPTrackingGlobal_Dpt50_v1', 
    'HLT_PADmesonPPTrackingGlobal_Dpt5_v1', 
    'HLT_PADmesonPPTrackingGlobal_Dpt8_v1')


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
    'HLT_PAL2DoubleMu10_v1', 
    'HLT_PAL2DoubleMuOpen_v1', 
    'HLT_PAL3DoubleMu10_v1', 
    'HLT_PAL3DoubleMuOpen_HIon_v1', 
    'HLT_PAL3DoubleMuOpen_v1')

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as streamPhysicsPAMuons_datasetPASingleMuon_selector
streamPhysicsPAMuons_datasetPASingleMuon_selector.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
streamPhysicsPAMuons_datasetPASingleMuon_selector.l1tResults = cms.InputTag('')
streamPhysicsPAMuons_datasetPASingleMuon_selector.throw      = cms.bool(False)
streamPhysicsPAMuons_datasetPASingleMuon_selector.triggerConditions = cms.vstring('HLT_PAAK4CaloJet40_Eta5p1_PAL3Mu3_v2', 
    'HLT_PAAK4CaloJet40_Eta5p1_PAL3Mu5_v2', 
    'HLT_PAAK4CaloJet60_Eta5p1_PAL3Mu3_v2', 
    'HLT_PAAK4CaloJet60_Eta5p1_PAL3Mu5_v2', 
    'HLT_PAL2Mu12_v1', 
    'HLT_PAL2Mu15_v1', 
    'HLT_PAL3Mu12_v1', 
    'HLT_PAL3Mu15_v1', 
    'HLT_PAL3Mu3_v1', 
    'HLT_PAL3Mu5_v1', 
    'HLT_PAL3Mu7_v1', 
    'HLT_PASinglePhoton10_Eta3p1_PAL3Mu3_v2', 
    'HLT_PASinglePhoton10_Eta3p1_PAL3Mu5_v2', 
    'HLT_PASinglePhoton15_Eta3p1_PAL3Mu3_v2', 
    'HLT_PASinglePhoton15_Eta3p1_PAL3Mu5_v2', 
    'HLT_PASinglePhoton20_Eta3p1_PAL3Mu3_v2', 
    'HLT_PASinglePhoton20_Eta3p1_PAL3Mu5_v2')

