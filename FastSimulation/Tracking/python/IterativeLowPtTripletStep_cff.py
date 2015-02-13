import FWCore.ParameterSet.Config as cms

# trajectory seeds
from RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi import PixelLayerTriplets
from FastSimulation.Tracking.TrajectorySeedProducer_cfi import trajectorySeedProducer
lowPtTripletStepSeeds = trajectorySeedProducer.clone(
    simTrackSelection = trajectorySeedProducer.simTrackSelection.clone(
        skipSimTrackIds = [    
            cms.InputTag("initialStepSimTrackIds"),
            cms.InputTag("detachedTripletStepSimTrackIds")],
        pTMin = 0.25,
        maxD0 = 5.,
        maxZ0 = 50.
        ),
    minLayersCrossed = 3,
    originRadius = 0.03,
    originHalfLength = 17.5,
    originpTMin = 0.35,
    layerList = PixelLayerTriplets.layerList
    )

# track candidates

from FastSimulation.Tracking.TrackCandidateProducer_cfi import trackCandidateProducer
lowPtTripletStepTrackCandidates = trackCandidateProducer.clone(
    SeedProducer = cms.InputTag("lowPtTripletStepSeeds"),
    MinNumberOfCrossedLayers = 3)

# track producer

from RecoTracker.IterativeTracking.LowPtTripletStep_cff import lowPtTripletStepTracks
lowPtTripletStepTracks = lowPtTripletStepTracks.clone(
    Fitter = 'KFFittingSmootherSecond',
    Propagator = 'PropagatorWithMaterial',
    TTRHBuilder = 'WithoutRefit'
)

# simtrack id producer
lowPtTripletStepSimTrackIds = cms.EDProducer("SimTrackIdProducer",
                                     trackCollection = cms.InputTag("lowPtTripletStepTracks"),
                                     HitProducer = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSMatchedRecHits")
                                     )

# TRACK SELECTION AND QUALITY FLAG SETTING.
from RecoTracker.IterativeTracking.LowPtTripletStep_cff import lowPtTripletStepSelector

LowPtTripletStep = cms.Sequence(lowPtTripletStepSeeds+
                                lowPtTripletStepTrackCandidates+
                                lowPtTripletStepTracks+  
                                lowPtTripletStepSimTrackIds+
                                lowPtTripletStepSelector)
