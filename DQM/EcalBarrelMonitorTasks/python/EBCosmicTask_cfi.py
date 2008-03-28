import FWCore.ParameterSet.Config as cms

ecalBarrelCosmicTask = cms.EDFilter("EBCosmicTask",
    EcalRawDataCollection = cms.InputTag("ecalEBunpacker"),
    EcalUncalibratedRecHitCollection = cms.InputTag("ecalUncalibHit","EcalUncalibRecHitsEB"),
    EcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    enableCleanup = cms.untracked.bool(True)
)


