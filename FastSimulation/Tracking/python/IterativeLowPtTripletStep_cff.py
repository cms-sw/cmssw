import FWCore.ParameterSet.Config as cms

# trajectory seeds

import FastSimulation.Tracking.TrajectorySeedProducer_cfi
iterativeLowPtTripletSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
iterativeLowPtTripletSeeds.simTrackSelection.skipSimTrackIds = [    
    cms.InputTag("initialStepSimTrackIds"),
    cms.InputTag("detachedTripletStepSimTrackIds")]
iterativeLowPtTripletSeeds.simTrackSelection.pTMin = 0.25
iterativeLowPtTripletSeeds.simTrackSelection.maxD0 = 5.
iterativeLowPtTripletSeeds.simTrackSelection.maxZ0 = 50.
iterativeLowPtTripletSeeds.minLayersCrossed = 3
iterativeLowPtTripletSeeds.originRadius = 0.03
iterativeLowPtTripletSeeds.originHalfLength = 17.5
iterativeLowPtTripletSeeds.originpTMin = 0.35
iterativeLowPtTripletSeeds.primaryVertex = ''
from RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi import PixelLayerTriplets
iterativeLowPtTripletSeeds.layerList = PixelLayerTriplets.layerList

# track candidates

from FastSimulation.Tracking.TrackCandidateProducer_cfi import trackCandidateProducer
lowPtTripletStepTrackCandidates = trackCandidateProducer.clone(
    SeedProducer = cms.InputTag("iterativeLowPtTripletSeeds"),
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

LowPtTripletStep = cms.Sequence(iterativeLowPtTripletSeeds+
                                lowPtTripletStepTrackCandidates+
                                lowPtTripletStepTracks+  
                                lowPtTripletStepSimTrackIds+
                                lowPtTripletStepSelector)

