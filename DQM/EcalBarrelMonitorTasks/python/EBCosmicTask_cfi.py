import FWCore.ParameterSet.Config as cms

ecalBarrelCosmicTask = cms.EDAnalyzer("EBCosmicTask",
    prefixME = cms.untracked.string('EcalBarrel'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),    
    EcalRawDataCollection = cms.InputTag("ecalEBunpacker"),
    EcalUncalibratedRecHitCollection = cms.InputTag("ecalUncalibHit","EcalUncalibRecHitsEB"),
    EcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB")
)

