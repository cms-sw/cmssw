import FWCore.ParameterSet.Config as cms

# trajectory seeds
from RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi import PixelLayerTriplets
from FastSimulation.Tracking.TrajectorySeedProducer_cfi import trajectorySeedProducer
initialStepSeeds = trajectorySeedProducer.clone(
    simTrackSelection = cms.PSet(
        pTMin = cms.double(0.4),
        maxD0 = cms.double(1.0),
        maxZ0 = cms.double(999),
        skipSimTrackIds = cms.VInputTag()
        ),
    minLayersCrossed = 3,
    # note: standard tracking uses for originRadius 0.03, but this value 
    # gives a much better agreement in rate and shape for iter0
    originpTMin = 0.6,
    originRadius = 1.0, 
    originHalfLength = 999,
    layerList = PixelLayerTriplets.layerList
)

# track candidates
from FastSimulation.Tracking.TrackCandidateProducer_cfi import trackCandidateProducer
initialStepTrackCandidates = trackCandidateProducer.clone(
    SeedProducer = cms.InputTag("initialStepSeeds"),
    MinNumberOfCrossedLayers = 3
)

# tracks
from RecoTracker.IterativeTracking.InitialStep_cff import initialStepTracks
initialStepTracks = initialStepTracks.clone(
    Fitter = 'KFFittingSmootherWithOutlierRejection',
    TTRHBuilder = 'WithoutRefit',
    Propagator = 'PropagatorWithMaterial')

# vertices
from RecoTracker.IterativeTracking.InitialStep_cff import firstStepPrimaryVertices
firstStepPrimaryVertices = firstStepPrimaryVertices.clone()

# simtrack id producer
initialStepSimTrackIds = cms.EDProducer("SimTrackIdProducer",
                                        trackCollection = cms.InputTag("initialStepTracks"),
                                        HitProducer = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSMatchedRecHits")
                                        )

# final selection
from RecoTracker.IterativeTracking.InitialStep_cff import initialStepSelector,initialStep

# Final sequence
InitialStep = cms.Sequence(initialStepSeeds
                           +initialStepTrackCandidates
                           +initialStepTracks                                    
                           +firstStepPrimaryVertices
                           +initialStepSelector
                           +initialStep
                           +initialStepSimTrackIds)

