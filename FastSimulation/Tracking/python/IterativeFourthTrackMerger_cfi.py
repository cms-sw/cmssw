import FWCore.ParameterSet.Config as cms

iterativeFourthTrackMerging = cms.EDFilter("FastTrackMerger",
    TrackProducers = cms.VInputTag(cms.InputTag("iterativeFourthTrackCandidatesWithPairs"),
                                   cms.InputTag("iterativeFourthTracksWithPairs")),
    trackAlgo = cms.untracked.uint32(4),
    MinNumberOfTrajHits = cms.untracked.uint32(5),
    MaxLostTrajHits = cms.untracked.uint32(0)
)


