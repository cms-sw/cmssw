import FWCore.ParameterSet.Config as cms

trackCandidateProducer = cms.EDProducer(
    "TrackCandidateProducer",
    recHits = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSMatchedRecHits"),
    # The smallest number of crossed layers to make a candidate
    MinNumberOfCrossedLayers = cms.uint32(5),

    src = cms.InputTag("globalPixelSeeds"),

    # Reject overlapping hits? (GroupedTracking from 170pre2 onwards)
    OverlapCleaning = cms.bool(False),

    # Split matched hits? 
    SplitHits = cms.bool(True),
    simTracks = cms.InputTag('famosSimHits'),
    
    propagator = cms.string('PropagatorWithMaterial')
)


