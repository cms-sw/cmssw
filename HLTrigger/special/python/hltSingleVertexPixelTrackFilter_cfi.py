import FWCore.ParameterSet.Config as cms

hltSingleVertexPixelTrackFilter = cms.EDFilter("HLTSingleVertexPixelTrackFilter",
    vertexCollection = cms.InputTag("hltPixelVerticesForMinBias"),
    trackCollection = cms.InputTag("hltPixelCands"),
    saveTags = cms.bool( False ),
    MinTrks = cms.int32(30),
    MaxEta = cms.double(1.0),
    MaxVz = cms.double(10.0),
    MinPt = cms.double(0.2),
    MaxPt = cms.double(10000.0),
    MinSep = cms.double(0.12)
)
