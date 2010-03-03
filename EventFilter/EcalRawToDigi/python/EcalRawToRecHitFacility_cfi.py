import FWCore.ParameterSet.Config as cms

EcalRawToRecHitFacility = cms.EDProducer("EcalRawToRecHitFacility",
    sourceTag = cms.InputTag("rawDataCollector"),
    workerName = cms.string('')
)


