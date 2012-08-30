import FWCore.ParameterSet.Config as cms

ecalBarrelCosmicTask = cms.EDAnalyzer("EBCosmicTask",
    prefixME = cms.untracked.string('EcalBarrel'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),    
    EcalRawDataCollection = cms.InputTag("ecalDigis"),
    EcalUncalibratedRecHitCollection = cms.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEB"),
    EcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB")
)

