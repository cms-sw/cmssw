import FWCore.ParameterSet.Config as cms

recoTrackSelector = cms.EDProducer("RecoTrackSelector",
    src = cms.InputTag("generalTracks"),
    maxChi2 = cms.double(10000.0),
    tip = cms.double(120.0),
    minRapidity = cms.double(-5.0),
    lip = cms.double(300.0),
    ptMin = cms.double(0.1),
    maxRapidity = cms.double(5.0),
    quality = cms.vstring('loose'),
    algorithm = cms.vstring(),
    minHit = cms.int32(3),
    min3DHit = cms.int32(0),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    copyExtras = cms.untracked.bool(True), ## copies also extras and rechits on RECO
    copyTrajectories = cms.untracked.bool(False) # don't set this to true on AOD!
)



