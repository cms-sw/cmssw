import FWCore.ParameterSet.Config as cms

pfPileUpJME = cms.EDProducer("PFPileUp",
    enable = cms.bool(True),
    PFCandidates = cms.InputTag("particleFlowPtrs"),
    Vertices = cms.InputTag("goodOfflinePrimaryVertices"),
    checkClosestZVertex = cms.bool(False),
    verbose = cms.untracked.bool(False),
    useVertexAssociation = cms.bool(False),
    vertexAssociation = cms.InputTag(""),
    vertexAssociationQuality = cms.int32(7)
)
