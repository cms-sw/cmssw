import FWCore.ParameterSet.Config as cms

hltL3TrackCandidateFromL2 = cms.EDProducer("TrackCandidateProducer",
    HitProducer = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSMatchedRecHits"),
    # The smallest number of crossed layers to make a candidate
    MinNumberOfCrossedLayers = cms.uint32(5),
    # The number of crossed layers needed before stopping tracking
    MaxNumberOfCrossedLayers = cms.uint32(999),
    SeedProducer = cms.InputTag("hltL3TrajectorySeed"),
    # If true, keep the tracks fitted in the previous collection
    # Just ignore them otherwise
    KeepFittedTracks = cms.bool(True),
    # Reject overlapping hits? (GroupedTracking from 170pre2 onwards)
    OverlapCleaning = cms.bool(False),
    # Reject copies of tracks from several seeds - take the first seed in that case
    SeedCleaning = cms.bool(True),
    # The tracks already fitted - no need to fit them already !
    TrackProducers = cms.VInputTag(),
    # Split matched hits? 
    SplitHits = cms.bool(True)
)


