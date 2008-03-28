import FWCore.ParameterSet.Config as cms

simpleSecondaryVertex = cms.ESProducer("SimpleSecondaryVertexESProducer",
    use3d = cms.bool(True),
    unBoost = cms.bool(False),
    useSignificance = cms.bool(True)
)


