import FWCore.ParameterSet.Config as cms

iterativeThirdTrackMerging = cms.EDFilter("FastTrackMerger",
    TrackProducers = cms.VInputTag(cms.InputTag("iterativeThirdTrackCandidatesWithPairs"),
                                   cms.InputTag("iterativeThirdTracksWithPairs")),
    trackAlgo = cms.untracked.uint32(3),
    MinNumberOfTrajHits = cms.untracked.uint32(3),
    MaxLostTrajHits = cms.untracked.uint32(0)                                          
)


