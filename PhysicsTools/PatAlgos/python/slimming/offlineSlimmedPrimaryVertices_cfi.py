import FWCore.ParameterSet.Config as cms

offlineSlimmedPrimaryVertices = cms.EDProducer("PATVertexSlimmer",
    src = cms.InputTag("offlinePrimaryVertices"),
)
