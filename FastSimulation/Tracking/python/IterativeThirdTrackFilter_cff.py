import FWCore.ParameterSet.Config as cms

thStep = cms.EDFilter("VertexFilter",
    TrackAlgorithm = cms.string('iter3'),
    recVertices = cms.InputTag("pixelVertices"),
    MinHits = cms.int32(3),
    DistRhoFromVertex = cms.double(0.1),
    DistZFromVertex = cms.double(0.1),
    recTracks = cms.InputTag("iterativeThirdTrackMerging"),
    UseQuality = cms.bool(True),
    ChiCut = cms.double(250000.0),
    TrackQuality = cms.string('highPurity'),
    VertexCut = cms.bool(True)
)

iterativeThirdTrackFiltering = cms.Sequence(thStep)

