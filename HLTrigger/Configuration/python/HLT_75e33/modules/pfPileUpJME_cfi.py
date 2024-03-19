import FWCore.ParameterSet.Config as cms

pfPileUpJME = cms.EDProducer("PFPileUp",
    PFCandidates = cms.InputTag("particleFlowPtrs"),
    Vertices = cms.InputTag("goodOfflinePrimaryVertices"),
    checkClosestZVertex = cms.bool(False),
    enable = cms.bool(True),
    useVertexAssociation = cms.bool(False),
    verbose = cms.untracked.bool(False),
    vertexAssociation = cms.InputTag(""),
    vertexAssociationQuality = cms.int32(7)
)
