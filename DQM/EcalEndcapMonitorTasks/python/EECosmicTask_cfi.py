import FWCore.ParameterSet.Config as cms

ecalEndcapCosmicTask = cms.EDAnalyzer("EECosmicTask",
    prefixME = cms.untracked.string('EcalEndcap'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),    
    EcalRawDataCollection = cms.InputTag("ecalDigis"),
    EcalUncalibratedRecHitCollection = cms.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEE"),
    EcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE")
)

