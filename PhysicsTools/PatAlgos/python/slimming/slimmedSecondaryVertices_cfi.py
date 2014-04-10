import FWCore.ParameterSet.Config as cms

slimmedSecondaryVertices = cms.EDProducer("PATSecondaryVertexSlimmer",
    src = cms.InputTag("inclusiveSecondaryVertices"),
    map = cms.InputTag("packedPFCandidates"),
)
