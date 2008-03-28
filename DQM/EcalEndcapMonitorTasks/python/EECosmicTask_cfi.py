import FWCore.ParameterSet.Config as cms

ecalEndcapCosmicTask = cms.EDFilter("EECosmicTask",
    EcalRawDataCollection = cms.InputTag("ecalEBunpacker"),
    EcalUncalibratedRecHitCollection = cms.InputTag("ecalUncalibHit","EcalUncalibRecHitsEE"),
    EcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    enableCleanup = cms.untracked.bool(True)
)


