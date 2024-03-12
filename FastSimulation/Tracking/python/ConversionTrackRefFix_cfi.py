import FWCore.ParameterSet.Config as cms

fixedConversionTracks = cms.EDProducer(
    "ConversionTrackRefFix",
    src = cms.InputTag("generalConversionTrackProducer"),
    newTrackCollection = cms.InputTag("generalTracks")
    )
# foo bar baz
# BCm6UVefRCz0c
