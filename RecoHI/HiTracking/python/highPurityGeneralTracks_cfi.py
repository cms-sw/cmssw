import FWCore.ParameterSet.Config as cms

highPurityGeneralTracks = cms.EDFilter(
    'TrackSelector',
    src = cms.InputTag('generalTracks'),
    cut = cms.string('quality("highPurity")'),
    )
# foo bar baz
# nnoNExQZ9iu4o
