import FWCore.ParameterSet.Config as cms

ecalBarrelBeamCaloTask = cms.EDFilter("EBBeamCaloTask",
    EBDigiCollection = cms.InputTag("ecalEBunpacker","ebDigis"),
    EcalTBEventHeader = cms.InputTag("ecalEBunpacker"),
    EcalRawDataCollection = cms.InputTag("ecalEBunpacker"),
    EcalUncalibratedRecHitCollection = cms.InputTag("ecalUncalibHit","EcalUncalibRecHitsEB"),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),
    prefixME = cms.untracked.string('EcalBarrel')
)


