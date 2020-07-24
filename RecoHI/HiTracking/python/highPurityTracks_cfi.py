import FWCore.ParameterSet.Config as cms

highPurityTracks = cms.EDFilter(
    'TrackSelector',
    src = cms.InputTag('generalTracks'),
    cut = cms.string('quality("highPurity")'),
    )
