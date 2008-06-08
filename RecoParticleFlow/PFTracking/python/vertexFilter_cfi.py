import FWCore.ParameterSet.Config as cms

vertFilter = cms.EDFilter("VertexFilter",
    TrackAlgorithm = cms.string('iter2'),
    recVertices = cms.InputTag("pixelVertices"),
    MinHits = cms.int32(3),
    TrackQualities = cms.vstring('loose', 
        'tight', 
        'highPurity'),
    recTracks = cms.InputTag("secWithMaterialTracks"),
    DistZFromVertex = cms.double(0.4),
    DistRhoFromVertex = cms.double(0.1),
    UseQuality = cms.bool(True),
    ChiCut = cms.double(130.0),
    VertexCut = cms.bool(True)
)



