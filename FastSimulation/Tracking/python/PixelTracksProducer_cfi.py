import FWCore.ParameterSet.Config as cms

pixelTracks = cms.EDFilter("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("hltPixelTracks"))
)


