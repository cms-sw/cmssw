import FWCore.ParameterSet.Config as cms

highPurityGeneralTracks = cms.EDFilter(
    'TrackSelector',
    src = cms.InputTag('generalTracks'),
    cut = cms.string('quality("highPurity")'),
    )
