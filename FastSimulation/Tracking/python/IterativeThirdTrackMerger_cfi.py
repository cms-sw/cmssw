import FWCore.ParameterSet.Config as cms

iterativeThirdTrackMerging = cms.EDFilter("FastTrackMerger",
    TrackProducers = cms.VInputTag(cms.InputTag("iterativeThirdTrackCandidatesWithPairs"), cms.InputTag("iterativeThirdTracksWithPairs"))
)


