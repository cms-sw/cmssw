import FWCore.ParameterSet.Config as cms

iterativeFourthTrackMerging = cms.EDFilter("FastTrackMerger",
    TrackProducers = cms.VInputTag(cms.InputTag("iterativeFourthTrackCandidatesWithPairs"),
                                   cms.InputTag("iterativeFourthTracksWithPairs")),
    trackAlgo = cms.untracked.uint32(4)
)


