import FWCore.ParameterSet.Config as cms

gemDigiToRaw = cms.EDProducer("GEMDigiToRawModule",
    gemDigi = cms.InputTag("simGEMDigis"),
    eventType = cms.Int(0),
)


