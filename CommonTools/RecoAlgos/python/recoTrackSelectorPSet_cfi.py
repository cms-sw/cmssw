import FWCore.ParameterSet.Config as cms

recoTrackSelectorPSet = cms.PSet(
    src = cms.InputTag("generalTracks"),
    maxChi2 = cms.double(10000.0),
    tip = cms.double(120.0),
    minRapidity = cms.double(-5.0),
    lip = cms.double(300.0),
    ptMin = cms.double(0.1),
    maxRapidity = cms.double(5.0),
    quality = cms.vstring('loose'),
    algorithm = cms.vstring(),
    originalAlgorithm = cms.vstring(),
    algorithmMaskContains = cms.vstring(),
    minLayer = cms.int32(3),
    min3DLayer = cms.int32(0),
    minHit = cms.int32(0),
    minPixelHit = cms.int32(0),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    usePV = cms.bool(False),
    vertexTag = cms.InputTag('offlinePrimaryVertices')
)
