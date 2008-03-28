import FWCore.ParameterSet.Config as cms

firstfilter = cms.EDFilter("QualityFilter",
    TrackQuality = cms.string('highPurity'),
    recTracks = cms.InputTag("generalTracks")
)


