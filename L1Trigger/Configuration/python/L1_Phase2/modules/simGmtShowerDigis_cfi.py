import FWCore.ParameterSet.Config as cms

simGmtShowerDigis = cms.EDProducer("L1TMuonShowerProducer",
    bxMax = cms.int32(0),
    bxMin = cms.int32(0),
    showerInput = cms.InputTag("simEmtfShowers","EMTF")
)
