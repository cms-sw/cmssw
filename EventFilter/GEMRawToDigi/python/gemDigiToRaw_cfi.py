import FWCore.ParameterSet.Config as cms

gempacker = cms.EDProducer("GEMDigiToRawModule",
    gemDigi = cms.InputTag("simMuonGEMDigis"),
    eventType = cms.Int(0),
)
