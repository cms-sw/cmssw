import FWCore.ParameterSet.Config as cms

# trajectory seeds

import FastSimulation.Tracking.TrajectorySeedProducer_cfi
iterativePixelLessSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
iterativePixelLessSeeds.simTrackSelection.skipSimTrackIds = [
    cms.InputTag("initialStepSimTrackIds"), 
    cms.InputTag("detachedTripletStepSimTrackIds"), 
    cms.InputTag("lowPtTripletStepSimTrackIds"), 
    cms.InputTag("pixelPairStepSimTrackIds"),  
    cms.InputTag("mixedTripletStepSimTrackIds")]
iterativePixelLessSeeds.simTrackSelection.pTMin = 0.3
iterativePixelLessSeeds.simTrackSelection.maxD0 = 99.
iterativePixelLessSeeds.simTrackSelection.maxZ0 = 99.
iterativePixelLessSeeds.minLayersCrossed = 3
iterativePixelLessSeeds.originRadius = 1.0
iterativePixelLessSeeds.originHalfLength = 12.0
iterativePixelLessSeeds.originpTMin = 0.4
iterativePixelLessSeeds.primaryVertex = ''
from RecoTracker.IterativeTracking.PixelLessStep_cff import pixelLessStepSeedLayers
iterativePixelLessSeeds.layerList = pixelLessStepSeedLayers.layerList

# candidate producer
from FastSimulation.Tracking.TrackCandidateProducer_cfi import trackCandidateProducer
pixelLessStepTrackCandidates = trackCandidateProducer.clone(
    SeedProducer = cms.InputTag("iterativePixelPairSeeds"),
    MinNumberOfCrossedLayers = 6 # ?
)

# track producer
from RecoTracker.IterativeTracking.PixelLessStep_cff import pixelLessStepTracks
pixelLessStepTracks = pixelLessStepTracks.clone(
    TTRHBuilder = 'WithoutRefit',
    Fitter = 'KFFittingSmootherFourth',
    Propagator = 'PropagatorWithMaterial'
)

# simtrack id producer
pixelLessStepSimTrackIds = cms.EDProducer("SimTrackIdProducer",
                                          trackCollection = cms.InputTag("pixelLessStepTracks"),
                                          HitProducer = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSMatchedRecHits")
                                          )

# track selection
from RecoTracker.IterativeTracking.PixelLessStep_cff import pixelLessStepSelector,pixelLessStep

# simtrack id producer
pixelLessStepIds = cms.EDProducer("SimTrackIdProducer",
                                  trackCollection = cms.InputTag("iterativePixelLessTracks"),
                                  HitProducer = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSMatchedRecHits")
                                  )

# sequence
PixelLessStep = cms.Sequence(iterativePixelLessSeeds+
                             pixelLessStepTrackCandidates+
                             pixelLessStepTracks+
                             pixelLessStepSimTrackIds+
                             pixelLessStepSelector+
                             pixelLessStep)
