import FWCore.ParameterSet.Config as cms

ecalEndcapCosmicTask = cms.EDFilter("EECosmicTask",
    EcalRawDataCollection = cms.InputTag("ecalEBunpacker"),
    EcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    enableCleanup = cms.untracked.bool(False),
    prefixME = cms.untracked.string('EcalEndcap'),
    EcalUncalibratedRecHitCollection = cms.InputTag("ecalUncalibHit","EcalUncalibRecHitsEE")
)


