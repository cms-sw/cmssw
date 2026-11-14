import FWCore.ParameterSet.Config as cms

simMuonGEMPadDigis = cms.EDProducer("GEMPadDigiProducer",
    InputCollection = cms.InputTag("simMuonGEMDigis")
)
