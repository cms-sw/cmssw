import FWCore.ParameterSet.Config as cms

softPFMuonsTagInfos = cms.EDProducer("SoftPFMuonTagInfoProducer",
    primaryVertex = cms.InputTag("offlinePrimaryVertices"),
    jets = cms.InputTag("ak5PFJets"),
    MuonId =cms.int32(0)
)
