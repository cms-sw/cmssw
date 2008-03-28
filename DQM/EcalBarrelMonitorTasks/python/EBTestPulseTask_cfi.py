import FWCore.ParameterSet.Config as cms

ecalBarrelTestPulseTask = cms.EDFilter("EBTestPulseTask",
    EBDigiCollection = cms.InputTag("ecalEBunpacker","ebDigis"),
    EcalPnDiodeDigiCollection = cms.InputTag("ecalEBunpacker"),
    EcalUncalibratedRecHitCollection = cms.InputTag("ecalUncalibHit","EcalUncalibRecHitsEB"),
    EcalRawDataCollection = cms.InputTag("ecalEBunpacker"),
    enableCleanup = cms.untracked.bool(True)
)


