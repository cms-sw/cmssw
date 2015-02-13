import FWCore.ParameterSet.Config as cms

# trajectory seeds

import FastSimulation.Tracking.TrajectorySeedProducer_cfi
iterativePixelPairSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
iterativePixelPairSeeds.simTrackSelection.skipSimTrackIds = [
    cms.InputTag("initialStepSimTrackIds"), 
    cms.InputTag("detachedTripletStepSimTrackIds"), 
    cms.InputTag("lowPtTripletStepSimTrackIds")]
iterativePixelPairSeeds.simTrackSelection.pTMin = 0.3
iterativePixelPairSeeds.simTrackSelection.maxD0 = 5.
iterativePixelPairSeeds.simTrackSelection.maxZ0 = 50.
iterativePixelPairSeeds.minLayersCrossed =3
iterativePixelPairSeeds.originRadius = 0.2
iterativePixelPairSeeds.originHalfLength = 17.5
iterativePixelPairSeeds.originpTMin = 0.6
iterativePixelPairSeeds.beamSpot = ''
iterativePixelPairSeeds.primaryVertex = 'firstStepPrimaryVertices' # vertices are generated from the initalStepTracks
from RecoTracker.IterativeTracking.PixelPairStep_cff import pixelPairStepSeedLayers
iterativePixelPairSeeds.layerList = pixelPairStepSeedLayers.layerList

# candidate producer
from FastSimulation.Tracking.TrackCandidateProducer_cfi import trackCandidateProducer
pixelPairStepTrackCandidates = trackCandidateProducer.clone(
    SeedProducer = cms.InputTag("iterativePixelPairSeeds"),
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
PixelPairStep = cms.Sequence(iterativePixelPairSeeds+
                             pixelPairStepTrackCandidates+
                             pixelPairStepTracks+
                             pixelPairStepSimTrackIds+
                             pixelPairStepSelector)
