import FWCore.ParameterSet.Config as cms

vertFilter = cms.EDFilter("VertexFilter",
    TrackAlgorithm = cms.string('iter2'),
    recVertices = cms.InputTag("pixelVertices"),
    MinHits = cms.int32(3),
    DistRhoFromVertex = cms.double(0.1),
    DistZFromVertex = cms.double(0.4),
    recTracks = cms.InputTag("secWithMaterialTracks"),
    UseQuality = cms.bool(True),
    ChiCut = cms.double(130.0),
    TrackQuality = cms.string('highPurity'),
    VertexCut = cms.bool(True)
)


