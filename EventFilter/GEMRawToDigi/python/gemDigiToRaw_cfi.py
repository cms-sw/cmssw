import FWCore.ParameterSet.Config as cms

gemRaw = cms.EDProducer("GEMDigiToRawModule",
    gemDigi = cms.InputTag("simGEMDigis"),
    eventType = cms.Int(0),
)


