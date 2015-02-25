import FWCore.ParameterSet.Config as cms

# trajectory seeds
from FastSimulation.Tracking.TrajectorySeedProducer_cfi import trajectorySeedProducer
from RecoTracker.IterativeTracking.PixelPairStep_cff import pixelPairStepSeedLayers
pixelPairStepSeeds = trajectorySeedProducer.clone(
    simTrackSelection = trajectorySeedProducer.simTrackSelection.clone(
        skipSimTrackIds = [
            cms.InputTag("initialStepSimTrackIds"), 
            cms.InputTag("detachedTripletStepSimTrackIds"), 
            cms.InputTag("lowPtTripletStepSimTrackIds")],
        pTMin = 0.3,
        maxD0 = 5.,
        maxZ0 = 50.
        ),
    minLayersCrossed =3,
    originRadius = 0.2,
    originHalfLength = 17.5,
    originpTMin = 0.6,
    beamSpot = '',
    primaryVertex = 'firstStepPrimaryVertices', # vertices are generated from the initalStepTracks
    layerList = pixelPairStepSeedLayers.layerList
)

# candidate producer
from FastSimulation.Tracking.TrackCandidateProducer_cfi import trackCandidateProducer
pixelPairStepTrackCandidates = trackCandidateProducer.clone(
    SeedProducer = cms.InputTag("pixelPairStepSeeds"),
    MinNumberOfCrossedLayers = 2 # ?
)

# track producer
from RecoTracker.IterativeTracking.PixelPairStep_cff import pixelPairStepTracks
pixelPairStepTracks = pixelPairStepTracks.clone(
    TTRHBuilder = 'WithoutRefit',
    Fitter = 'KFFittingSmootherSecond',
    Propagator = 'PropagatorWithMaterial',
)

# simtrack id producer
pixelPairStepSimTrackIds = cms.EDProducer("SimTrackIdProducer",
                                          trackCollection = cms.InputTag("pixelPairStepTracks"),
                                          HitProducer = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSMatchedRecHits")
                                          )

# Final selection
from RecoTracker.IterativeTracking.PixelPairStep_cff import pixelPairStepSelector


# sequence
PixelPairStep = cms.Sequence(pixelPairStepSeeds+
                             pixelPairStepTrackCandidates+
                             pixelPairStepTracks+
                             pixelPairStepSimTrackIds+
                             pixelPairStepSelector)
