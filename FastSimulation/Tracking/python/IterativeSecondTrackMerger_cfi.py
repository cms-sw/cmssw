import FWCore.ParameterSet.Config as cms

iterativeSecondTrackMerging = cms.EDProducer("FastTrackMerger",
    TrackProducers = cms.VInputTag(cms.InputTag("iterativeSecondTrackCandidatesWithTriplets"),
                                   cms.InputTag("iterativeSecondTracksWithTriplets")),
    RemoveTrackProducers =  cms.untracked.VInputTag(cms.InputTag("zeroStepFilter"), #prova
                                   cms.InputTag("firstfilter")),#prova
    trackAlgo = cms.untracked.uint32(6),
    MinNumberOfTrajHits = cms.untracked.uint32(3),
    MaxLostTrajHits = cms.untracked.uint32(1)
)



