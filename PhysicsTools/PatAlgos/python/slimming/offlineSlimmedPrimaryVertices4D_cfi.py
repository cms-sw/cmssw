import FWCore.ParameterSet.Config as cms

offlineSlimmedPrimaryVertices4D = cms.EDProducer("PATVertexSlimmer",
    src = cms.InputTag("offlinePrimaryVertices4D"),
)
# foo bar baz
# tKBl6pzyuvvkR
