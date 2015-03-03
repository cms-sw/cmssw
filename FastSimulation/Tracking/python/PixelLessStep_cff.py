import FWCore.ParameterSet.Config as cms

# import the full tracking equivalent of this file
import RecoTracker.IterativeTracking.PixelLessStep_cff

# trajectory seeds 
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
pixelLessStepSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    simTrackSelection = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.simTrackSelection.clone(
        skipSimTrackIds = [
            cms.InputTag("initialStepSimTrackIds"),
            cms.InputTag("detachedTripletStepSimTrackIds"),
            cms.InputTag("lowPtTripletStepSimTrackIds"),
            cms.InputTag("pixelPairStepSimTrackIds"),
            cms.InputTag("mixedTripletStepSimTrackIds")],
        pTMin = 0.3,
        maxD0 = 99,
        maxZ0 = 99
        ),
    minLayersCrossed = 3,
    originpTMin = RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepSeeds.RegionFactoryPSet.RegionPSet.ptMin,
    originHalfLength = RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepSeeds.RegionFactoryPSet.RegionPSet.originHalfLength,
    originRadius = RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepSeeds.RegionFactoryPSet.RegionPSet.originRadius,
    layerList = RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepSeedLayers.layerList.value()
)

# track candidates
import FastSimulation.Tracking.TrackCandidateProducer_cfi
pixelLessStepTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
    SeedProducer = cms.InputTag("pixelLessStepSeeds"),
    MinNumberOfCrossedLayers = 6 # ?
)

# tracks
pixelLessStepTracks = RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepTracks.clone(
    TTRHBuilder = 'WithoutRefit',
    Fitter = 'KFFittingSmootherFourth',
    Propagator = 'PropagatorWithMaterial'
)

# simtrack id producer
pixelLessStepSimTrackIds = cms.EDProducer("SimTrackIdProducer",
                                          trackCollection = cms.InputTag("pixelLessStepTracks"),
                                          HitProducer = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSMatchedRecHits")
)

# final selection
pixelLessStepSelector = RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepSelector.clone()
pixelLessStep = RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStep.clone()

# Final sequence 
PixelLessStep = cms.Sequence(pixelLessStepSeeds
                             +pixelLessStepTrackCandidates
                             +pixelLessStepTracks
                             +pixelLessStepSelector
                             +pixelLessStep
                             +pixelLessStepSimTrackIds
                         )

