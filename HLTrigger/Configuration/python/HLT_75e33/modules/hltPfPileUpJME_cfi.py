import FWCore.ParameterSet.Config as cms

hltPfPileUpJME = cms.EDProducer("PFPileUp",
    PFCandidates = cms.InputTag("hltParticleFlowPtrs"),
    Vertices = cms.InputTag("hltGoodOfflinePrimaryVertices"),
    checkClosestZVertex = cms.bool(False),
    enable = cms.bool(True),
    useVertexAssociation = cms.bool(False),
    verbose = cms.untracked.bool(False),
    vertexAssociation = cms.InputTag(""),
    vertexAssociationQuality = cms.int32(7)
)
