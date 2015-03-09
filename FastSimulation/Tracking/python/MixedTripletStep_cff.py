import FWCore.ParameterSet.Config as cms

# import the full tracking equivalent of this file
import RecoTracker.IterativeTracking.MixedTripletStep_cff

# trajectory seeds
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
mixedTripletStepSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    simTrackSelection = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.simTrackSelection.clone(
        skipSimTrackIds = [
            cms.InputTag("initialStepSimTrackIds"),
            cms.InputTag("detachedTripletStepSimTrackIds"),
            cms.InputTag("lowPtTripletStepSimTrackIds"),
            cms.InputTag("pixelPairStepSimTrackIds")],
        pTMin = 0.15,
        maxD0 = 10.0,
        maxZ0 = 30
        ),
    minLayersCrossed = 3,
    originpTMin =  RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSeedsA.RegionFactoryPSet.RegionPSet.ptMin,
    originRadius = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSeedsA.RegionFactoryPSet.RegionPSet.originRadius,
    originHalfLength = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSeedsA.RegionFactoryPSet.RegionPSet.originHalfLength,
    layerList = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSeedLayersA.layerList.value()+RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSeedLayersB.layerList.value()
)
# Fullsim has different cuts for A and B, here in FastSIm only A cuts are implemented 
# need to do something about this ??

# track candidates
import FastSimulation.Tracking.TrackCandidateProducer_cfi
mixedTripletStepTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
    SeedProducer = cms.InputTag("mixedTripletStepSeeds"),
    MinNumberOfCrossedLayers = 3
)

# tracks
mixedTripletStepTracks = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepTracks.clone(
    TTRHBuilder = 'WithoutRefit',
    Fitter = 'KFFittingSmootherThird',
    Propagator = 'PropagatorWithMaterial'
)

# simtrack id producer
mixedTripletStepSimTrackIds = cms.EDProducer("SimTrackIdProducer",
                                             trackCollection = cms.InputTag("mixedTripletStepTracks"),
                                             HitProducer = cms.InputTag("siTrackerGaussianSmearingRecHits","TrackerGSMatchedRecHits")
                                         )

# final selection
mixedTripletStepSelector = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSelector.clone()
mixedTripletStep = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStep.clone()

# Final sequence 
MixedTripletStep =  cms.Sequence(mixedTripletStepSeeds
                                 +mixedTripletStepTrackCandidates
                                 +mixedTripletStepTracks
                                 +mixedTripletStepSelector
                                 +mixedTripletStep
                                 +mixedTripletStepSimTrackIds
                             )
