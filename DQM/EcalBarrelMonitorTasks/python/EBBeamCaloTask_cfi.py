import FWCore.ParameterSet.Config as cms

ecalBarrelBeamCaloTask = cms.EDAnalyzer("EBBeamCaloTask",
    prefixME = cms.untracked.string('EcalBarrel'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),
    EcalRawDataCollection = cms.InputTag("ecalEBunpacker"),
    EcalTBEventHeader = cms.InputTag("ecalEBunpacker"),
    EBDigiCollection = cms.InputTag("ecalEBunpacker","ebDigis"),
    EcalUncalibratedRecHitCollection = cms.InputTag("ecalUncalibHit","EcalUncalibRecHitsEB")
)

