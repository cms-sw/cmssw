import FWCore.ParameterSet.Config as cms

simpleSecondaryVertex2TrkComputer = cms.ESProducer("SimpleSecondaryVertexESProducer",
    minTracks = cms.uint32(2),
    unBoost = cms.bool(False),
    use3d = cms.bool(True),
    useSignificance = cms.bool(True)
)
