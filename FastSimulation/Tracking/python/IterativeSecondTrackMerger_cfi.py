import FWCore.ParameterSet.Config as cms

iterativeSecondTrackMerging = cms.EDFilter("FastTrackMerger",
    TrackProducers = cms.VInputTag(cms.InputTag("iterativeSecondTrackCandidatesWithTriplets"), cms.InputTag("iterativeSecondTracksWithTriplets"), cms.InputTag("iterativeSecondTrackCandidatesWithPlTriplets"), cms.InputTag("iterativeSecondTracksWithPlTriplets"))
)


