import FWCore.ParameterSet.Config as cms

packedPFCandidates = cms.EDProducer("PATPackedCandidateProducer",
    inputCollection = cms.InputTag("particleFlow"),
    inputVertices = cms.InputTag("offlineSlimmedPrimaryVertices"),
    originalVertices = cms.InputTag("offlinePrimaryVertices"),
    originalTracks = cms.InputTag("generalTracks"),
    vertexAssociator = cms.InputTag("primaryVertexAssociation","original"),
    PuppiWeight = cms.InputTag("puppi","PuppiWeights"),
    PuppiSrc = cms.InputTag("puppi"),
    PuppiSrcMap = cms.InputTag("puppi"),
    minPtForTrackProperties = cms.double(0.95)
)
