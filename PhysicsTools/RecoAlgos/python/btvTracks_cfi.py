import FWCore.ParameterSet.Config as cms

_content = cms.PSet(
    src = cms.InputTag("generalTracks"),
    maxChi2 = cms.double(5.0),
    tip = cms.double(0.2),
    minRapidity = cms.double(-9.0),
    lip = cms.double(17.0),
    ptMin = cms.double(1.0),
    maxRapidity = cms.double(9.0),
    quality = cms.vstring(),
    algorithm = cms.vstring(),
    minLayer = cms.int32(0),
    min3DLayer = cms.int32(0),
    minHit = cms.int32(8),
    minPixelHit = cms.int32(2),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    usePV = cms.bool(True),
    vertexTag = cms.InputTag('offlinePrimaryVertices'),
)

btvTracks = cms.EDProducer("RecoTrackSelector",
    _content,
    copyExtras = cms.untracked.bool(True), ## copies also extras and rechits on RECO
    copyTrajectories = cms.untracked.bool(False) # don't set this to true on AOD!
)

btvTrackRefs = cms.EDFilter("RecoTrackRefSelector",
    _content
)


