import FWCore.ParameterSet.Config as cms

recoTrackSelector = cms.EDFilter("RecoTrackSelector",
    src = cms.InputTag("generalTracks"),
    algorithm = cms.string(''),
    maxChi2 = cms.double(10000.0),
    tip = cms.double(120.0),
    minRapidity = cms.double(-5.0),
    lip = cms.double(300.0),
    ptMin = cms.double(0.1),
    maxRapidity = cms.double(5.0),
    quality = cms.string('loose'),
    minHit = cms.int32(3)
)



