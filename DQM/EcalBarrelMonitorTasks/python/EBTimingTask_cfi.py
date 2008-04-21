import FWCore.ParameterSet.Config as cms

ecalBarrelTimingTask = cms.EDFilter("EBTimingTask",
    EcalRawDataCollection = cms.InputTag("ecalEBunpacker"),
    EcalUncalibratedRecHitCollection = cms.InputTag("ecalUncalibHit","EcalUncalibRecHitsEB"),
    enableCleanup = cms.untracked.bool(False),
    prefixME = cms.untracked.string('EcalBarrel')
)


