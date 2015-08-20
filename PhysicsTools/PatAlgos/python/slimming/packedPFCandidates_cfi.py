import FWCore.ParameterSet.Config as cms

packedPFCandidates = cms.EDProducer("PATPackedCandidateProducer",
    inputCollection = cms.InputTag("particleFlow"),
    inputVertices = cms.InputTag("offlineSlimmedPrimaryVertices"),
    originalVertices = cms.InputTag("offlinePrimaryVertices"),
    originalTracks = cms.InputTag("generalTracks"),
    vertexAssociator = cms.InputTag("primaryVertexAssociation","original"),
    PuppiSrc = cms.InputTag("puppi"),
    PuppiNoLepSrc = cms.InputTag("puppiNoLep"),    
    secondaryVerticesForWhiteList = cms.InputTag("inclusiveCandidateSecondaryVertices"),
    minPtForTrackProperties = cms.double(0.95)
)
