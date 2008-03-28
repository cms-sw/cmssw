import FWCore.ParameterSet.Config as cms

EcalRawToRecHitFacility = cms.EDFilter("EcalRawToRecHitFacility",
    sourceTag = cms.InputTag("rawDataCollector"),
    workerName = cms.string('')
)


