import FWCore.ParameterSet.Config as cms

# seeding

import FastSimulation.Tracking.TrajectorySeedProducer_cfi
iterativeDetachedTripletSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
iterativeDetachedTripletSeeds.simTrackSelection.skipSimTrackIds = [cms.InputTag("initialStepSimTrackIds")]
iterativeDetachedTripletSeeds.simTrackSelection.pTMin = 0.020
iterativeDetachedTripletSeeds.simTrackSelection.maxD0 = 30.
iterativeDetachedTripletSeeds.simTrackSelection.maxZ0 = 50.
iterativeDetachedTripletSeeds.minLayersCrossed = 3
iterativeDetachedTripletSeeds.originpTMin = 0.075
iterativeDetachedTripletSeeds.originRadius = 1.5
iterativeDetachedTripletSeeds.originHalfLength = 15.
iterativeDetachedTripletSeeds.primaryVertex = ''
from RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi import PixelLayerTriplets
iterativeDetachedTripletSeeds.layerList = PixelLayerTriplets.layerList

# track candidates
import FastSimulation.Tracking.TrackCandidateProducer_cfi
detachedTripletStepTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
    SeedProducer = cms.InputTag("iterativeDetachedTripletSeeds"),
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
DetachedTripletStep = cms.Sequence(iterativeDetachedTripletSeeds+
                                   detachedTripletStepTrackCandidates+
                                   detachedTripletStepTracks+
                                   detachedTripletStepSimTrackIds+
                                   detachedTripletStepSelector+
                                   detachedTripletStep)
