import FWCore.ParameterSet.Config as cms

ecalEndcapLedTask = cms.EDFilter("EELedTask",
    prefixME = cms.untracked.string('EcalEndcap'),
    EEDigiCollection = cms.InputTag("ecalEBunpacker","eeDigis"),
    EcalRawDataCollection = cms.InputTag("ecalEBunpacker"),
    EcalUncalibratedRecHitCollection = cms.InputTag("ecalUncalibHit","EcalUncalibRecHitsEE"),
    enableCleanup = cms.untracked.bool(False),
    EcalPnDiodeDigiCollection = cms.InputTag("ecalEBunpacker")
)


