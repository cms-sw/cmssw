import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
DelayedJetMETTrigger = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
DelayedJetMETTrigger.TriggerResultsTag = cms.InputTag( "TriggerResults", "", "HLT" )
DelayedJetMETTrigger.HLTPaths = cms.vstring(
    "HLT_PFMET120_PFMHT120_IDTight_v*"
)
DelayedJetMETTrigger.throw = False
DelayedJetMETTrigger.andOr = True


caloJetTimingProducerSingle = cms.EDProducer( "HLTCaloJetTimingProducer",
    jets = cms.InputTag( "ak4CaloJets" ),
    barrelJets = cms.bool( True ),
    endcapJets = cms.bool( False ),
    ecalCellEnergyThresh = cms.double( 0.5 ),
    ecalCellTimeThresh = cms.double( 12.5 ),
    ecalCellTimeErrorThresh = cms.double( 100.0 ),
    matchingRadius = cms.double( 0.4 ),
    ebRecHitsColl = cms.InputTag( 'ecalRecHit','EcalRecHitsEB' ),
    eeRecHitsColl = cms.InputTag( 'ecalRecHit','EcalRecHitsEE' )
)


delayedJetSelection = cms.EDFilter( "HLTCaloJetTimingFilter",
    saveTags = cms.bool( True ),
    jets = cms.InputTag( "ak4CaloJets" ),
    jetTimes = cms.InputTag( "caloJetTimingProducerSingle" ),
    jetCellsForTiming = cms.InputTag( 'caloJetTimingProducerSingle','jetCellsForTiming' ),
    jetEcalEtForTiming = cms.InputTag( 'caloJetTimingProducerSingle','jetEcalEtForTiming' ),
    minJets = cms.uint32( 1 ),
    jetTimeThresh = cms.double( 1.0 ),
    jetCellsForTimingThresh = cms.uint32( 5 ),
    jetEcalEtForTimingThresh = cms.double( 10.0 ),
    minJetPt = cms.double( 40.0 )
)

EXODelayedJetMETSkimSequence = cms.Sequence(
    DelayedJetMETTrigger * caloJetTimingProducerSingle * delayedJetSelection
)
