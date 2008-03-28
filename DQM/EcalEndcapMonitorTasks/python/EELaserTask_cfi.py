import FWCore.ParameterSet.Config as cms

ecalEndcapLaserTask = cms.EDFilter("EELaserTask",
    EcalPnDiodeDigiCollection = cms.InputTag("ecalEBunpacker"),
    EcalUncalibratedRecHitCollection = cms.InputTag("ecalUncalibHit","EcalUncalibRecHitsEE"),
    EEDigiCollection = cms.InputTag("ecalEBunpacker","eeDigis"),
    EcalRawDataCollection = cms.InputTag("ecalEBunpacker"),
    enableCleanup = cms.untracked.bool(True)
)


