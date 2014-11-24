import FWCore.ParameterSet.Config as cms

# good track selection

hiTracks = cms.EDFilter("TrackSelector",
                                src = cms.InputTag("hiGeneralTracks"),
                                cut = cms.string(
    'quality("highPurity")')
                                )


