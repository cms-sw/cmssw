import FWCore.ParameterSet.Config as cms

ecalBarrelTimingTask = cms.EDFilter("EBTimingTask",
    EcalUncalibratedRecHitCollection = cms.InputTag("ecalUncalibHit","EcalUncalibRecHitsEB"),
    enableCleanup = cms.untracked.bool(True)
)


