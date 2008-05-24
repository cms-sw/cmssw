import FWCore.ParameterSet.Config as cms

simMuonCSCSuppressedDigis = cms.EDProducer("CSCDigiSuppressor",
    stripDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCStripDigi"),
    lctTag = cms.InputTag("simCscTriggerPrimitiveDigis")
)

