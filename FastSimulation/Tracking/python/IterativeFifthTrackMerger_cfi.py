import FWCore.ParameterSet.Config as cms

iterativeFifthTrackMerging = cms.EDFilter("FastTrackMerger",
    TrackProducers = cms.VInputTag(cms.InputTag("iterativeFifthTrackCandidatesWithPairs"),
                                   cms.InputTag("iterativeFifthTracksWithPairs")),
    trackAlgo = cms.untracked.uint32(5),
    MinNumberOfTrajHits = cms.untracked.uint32(4),
    MaxLostTrajHits = cms.untracked.uint32(0)
                                          
)


