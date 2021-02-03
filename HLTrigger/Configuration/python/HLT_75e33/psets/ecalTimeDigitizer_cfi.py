import FWCore.ParameterSet.Config as cms

ecalTimeDigitizer = cms.PSet(
    EBtimeDigiCollection = cms.string('EBTimeDigi'),
    EEtimeDigiCollection = cms.string('EETimeDigi'),
    accumulatorType = cms.string('EcalTimeDigiProducer'),
    hitsProducerEB = cms.InputTag("g4SimHits","EcalHitsEB"),
    hitsProducerEE = cms.InputTag("g4SimHits","EcalHitsEE"),
    timeLayerBarrel = cms.int32(7),
    timeLayerEndcap = cms.int32(3)
)