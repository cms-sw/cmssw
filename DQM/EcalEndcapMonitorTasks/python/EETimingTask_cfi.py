import FWCore.ParameterSet.Config as cms

ecalEndcapTimingTask = cms.EDFilter("EETimingTask",
    EcalUncalibratedRecHitCollection = cms.InputTag("ecalUncalibHit","EcalUncalibRecHitsEE"),
    enableCleanup = cms.untracked.bool(True)
)


