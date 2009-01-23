import FWCore.ParameterSet.Config as cms

iterativeFirstTrackMerging = cms.EDFilter("FastTrackMerger",
    TrackProducers = cms.VInputTag(cms.InputTag("iterativeFirstTrackCandidatesWithTriplets"),
                                   cms.InputTag("iterativeFirstTracksWithTriplets"),
                                   cms.InputTag("iterativeFirstTrackCandidatesWithPairs"),
                                   cms.InputTag("iterativeFirstTracksWithPairs")),
    trackAlgo = cms.untracked.uint32(1)
)


