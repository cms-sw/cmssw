import FWCore.ParameterSet.Config as cms

# import the full tracking equivalent of this file
import RecoTracker.IterativeTracking.PixelPairStep_cff

# trajectory seeds
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
pixelPairStepSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    simTrackSelection = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.simTrackSelection.clone(
        skipSimTrackIds = [
            cms.InputTag("initialStepSimTrackIds"),
            cms.InputTag("detachedTripletStepSimTrackIds"),
            cms.InputTag("lowPtTripletStepSimTrackIds")],
        pTMin = 0.3,
        maxD0 = 5.0,
        maxZ0 = 50
        ),
    minLayersCrossed = 2,
    nSigmaZ = 3, 
    originpTMin = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepSeeds.RegionFactoryPSet.RegionPSet.ptMin,
    originRadius = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepSeeds.RegionFactoryPSet.RegionPSet.originRadius,
    layerList = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepSeedLayers.layerList.value()
)

# track candidate 
import FastSimulation.Tracking.TrackCandidateProducer_cfi
pixelPairStepTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
    SeedProducer = cms.InputTag("pixelPairStepSeeds"),
    MinNumberOfCrossedLayers = 2 # ?
)

# tracks
pixelPairStepTracks = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepTracks.clone(
    Fitter = 'KFFittingSmootherSecond',
    TTRHBuilder = 'WithoutRefit',
    Propagator = 'PropagatorWithMaterial'
)

# simtrack id producer
pixelPairStepSimTrackIds = cms.EDProducer("SimTrackIdProducer",
                                          trackCollection = cms.InputTag("pixelPairStepTracks"),
                                          HitProducer = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSMatchedRecHits")
                                          )

# final Selection
pixelPairStepSelector = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepSelector.clone()
#PixelPairStep = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStep.clone()

# Final sequence 
PixelPairStep = cms.Sequence(pixelPairStepSeeds
                             +pixelPairStepTrackCandidates
                             +pixelPairStepTracks
                             +pixelPairStepSelector                           
                             +pixelPairStepSimTrackIds
                         )
