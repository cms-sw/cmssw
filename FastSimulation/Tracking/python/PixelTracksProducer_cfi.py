import FWCore.ParameterSet.Config as cms

pixelTracks = cms.EDProducer("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("hltPixelTracks"))
)


