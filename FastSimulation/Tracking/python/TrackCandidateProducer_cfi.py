import FWCore.ParameterSet.Config as cms

trackCandidateProducer = cms.EDProducer("TrackCandidateProducer",
    HitProducer = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSMatchedRecHits"),
    TrackProducer = cms.InputTag("generalTracks"),
    # The smallest number of crossed layers to make a candidate
    MinNumberOfCrossedLayers = cms.uint32(5),
    # The number of crossed layers needed before stopping tracking
    MaxNumberOfCrossedLayers = cms.uint32(999),
    SeedProducer = cms.InputTag("globalPixelGSSeeds","GlobalPixel"),
    # Reject overlapping hits? (GroupedTracking from 170pre2 onwards)
    OverlapCleaning = cms.bool(False),
    # Reject copies of tracks from several seeds - take the first seed in that case
    SeedCleaning = cms.bool(True),
    # Split matched hits? 
    SplitHits = cms.bool(True)
)


