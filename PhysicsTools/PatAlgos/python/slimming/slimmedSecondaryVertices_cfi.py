import FWCore.ParameterSet.Config as cms

slimmedSecondaryVertices = cms.EDProducer("PATSecondaryVertexSlimmer",
    src = cms.InputTag("inclusiveMergedVertices"),
    packedPFCandidates = cms.InputTag("packedPFCandidates"),
    lostTracksCandidates = cms.InputTag("lostTracks"),
)
