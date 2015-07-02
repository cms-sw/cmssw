import FWCore.ParameterSet.Config as cms

# import the full tracking equivalent of this file
import RecoTracker.IterativeTracking.MixedTripletStep_cff

# fast tracking mask producer                                                                                                                                                                                                                                        
from FastSimulation.Tracking.FastTrackingMaskProducer_cfi import fastTrackingMaskProducer as _fastTrackingMaskProducer
mixedTripletStepFastTrackingMasks = _fastTrackingMaskProducer.clone(
    trackCollection = cms.InputTag("pixelPairStepTracks"),
    TrackQuality = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepClusters.TrackQuality,
    overrideTrkQuals = cms.InputTag('pixelPairStepSelector','pixelPairStep')  
)

# trajectory seeds
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
mixedTripletStepSeedsA = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    simTrackSelection = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.simTrackSelection.clone(
        pTMin = 0.15,
        maxD0 = 10.0,
        maxZ0 = 30
        ),
    minLayersCrossed = 3,
    hitMasks = cms.InputTag("mixedTripletStepFastTrackingMasks","hitMasks"),
    hitCombinationMasks = cms.InputTag("mixedTripletStepFastTrackingMasks","hitCombinationMasks"),
    ptMin =  RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSeedsA.RegionFactoryPSet.RegionPSet.ptMin,
    originRadius = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSeedsA.RegionFactoryPSet.RegionPSet.originRadius,
    originHalfLength = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSeedsA.RegionFactoryPSet.RegionPSet.originHalfLength,
    layerList = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSeedLayersA.layerList.value()
)

###
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
mixedTripletStepSeedsB = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    simTrackSelection = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.simTrackSelection.clone(
        pTMin = 0.15,
        maxD0 = 10.0,
        maxZ0 = 30
        ),
    minLayersCrossed = 3,
    hitMasks = cms.InputTag("mixedTripletStepFastTrackingMasks","hitMasks"),
    hitCombinationMasks = cms.InputTag("mixedTripletStepFastTrackingMasks","hitCombinationMasks"),
    ptMin =  RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSeedsB.RegionFactoryPSet.RegionPSet.ptMin,
    originRadius = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSeedsB.RegionFactoryPSet.RegionPSet.originRadius,
    originHalfLength = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSeedsB.RegionFactoryPSet.RegionPSet.originHalfLength,
    layerList = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSeedLayersB.layerList.value()
)

mixedTripletStepSeeds = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSeeds.clone()

#track candidates
import FastSimulation.Tracking.TrackCandidateProducer_cfi
mixedTripletStepTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
    src = cms.InputTag("mixedTripletStepSeeds"),
    MinNumberOfCrossedLayers = 3
)

# tracks
mixedTripletStepTracks = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepTracks.clone(
    TTRHBuilder = 'WithoutRefit',
    Fitter = 'KFFittingSmootherThird',
    Propagator = 'PropagatorWithMaterial'
)

# final selection
mixedTripletStepSelector = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSelector.clone()
mixedTripletStepSelector.vertices = "firstStepPrimaryVerticesBeforeMixing"
mixedTripletStep = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStep.clone()

# Final sequence 
MixedTripletStep =  cms.Sequence(mixedTripletStepFastTrackingMasks
                                 +mixedTripletStepSeedsA
                                 +mixedTripletStepSeedsB
                                 +mixedTripletStepSeeds
                                 +mixedTripletStepTrackCandidates
                                 +mixedTripletStepTracks
                                 +mixedTripletStepSelector
                                 +mixedTripletStep                                 
                             )
