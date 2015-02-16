import FWCore.ParameterSet.Config as cms

# seeding
from RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi import PixelLayerTriplets
from FastSimulation.Tracking.TrajectorySeedProducer_cfi import trajectorySeedProducer
detachedTripletStepSeeds = trajectorySeedProducer.clone(
    simTrackSelection = trajectorySeedProducer.simTrackSelection.clone(
        skipSimTrackIds = [cms.InputTag("initialStepSimTrackIds")],
        pTMin = 0.020,
        maxD0 = 30.,
        maxZ0 = 50.
        ),
    minLayersCrossed = 3,
    originpTMin = 0.075,
    originRadius = 1.5,
    originHalfLength = 15.,
    primaryVertex = '',
    layerList = PixelLayerTriplets.layerList
    )

# track candidates
import FastSimulation.Tracking.TrackCandidateProducer_cfi
detachedTripletStepTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
    SeedProducer = cms.InputTag("detachedTripletStepSeeds"),
    MinNumberOfCrossedLayers = 3)

# track producer
from RecoTracker.IterativeTracking.DetachedTripletStep_cff import detachedTripletStepTracks
detachedTripletStepTracks = detachedTripletStepTracks.clone(
    Fitter = 'KFFittingSmootherSecond',
    Propagator = 'PropagatorWithMaterial',
    TTRHBuilder = 'WithoutRefit'
)

# simtrack id producer
detachedTripletStepSimTrackIds = cms.EDProducer("SimTrackIdProducer",
                                  trackCollection = cms.InputTag("detachedTripletStepTracks"),
                                  HitProducer = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSMatchedRecHits")
                                  )


# TRACK SELECTION AND QUALITY FLAG SETTING.
from RecoTracker.IterativeTracking.DetachedTripletStep_cff import detachedTripletStepSelector,detachedTripletStep

# sequence
DetachedTripletStep = cms.Sequence(detachedTripletStepSeeds+
                                   detachedTripletStepTrackCandidates+
                                   detachedTripletStepTracks+
                                   detachedTripletStepSimTrackIds+
                                   detachedTripletStepSelector+
                                   detachedTripletStep)
