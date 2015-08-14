import FWCore.ParameterSet.Config as cms

# import the full tracking equivalent of this file
import RecoTracker.IterativeTracking.MixedTripletStep_cff

# fast tracking mask producer                                                                                                                                                                                                                                        
from FastSimulation.Tracking.FastTrackingMaskProducer_cfi import fastTrackingMaskProducer as _fastTrackingMaskProducer
mixedTripletStepMasks = _fastTrackingMaskProducer.clone(
    trackCollection = cms.InputTag("pixelPairStepTracks"),
    TrackQuality = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepClusters.TrackQuality,
    overrideTrkQuals = cms.InputTag('pixelPairStep',"QualityMasks"),                        
    oldHitCombinationMasks = cms.InputTag("pixelPairStepMasks","hitCombinationMasks"),
    oldHitMasks = cms.InputTag("pixelPairStepMasks","hitMasks")
)

# trajectory seeds
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
mixedTripletStepSeedsA = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    simTrackSelection = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.simTrackSelection.clone(
        pTMin = 0,
        maxD0 = -1,
        maxZ0 = -1
        ),
    minLayersCrossed = 3,
layerList = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSeedLayersA.layerList.value(),
    RegionFactoryPSet = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSeedsA.RegionFactoryPSet,
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
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
layerList = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSeedLayersB.layerList.value(),
    RegionFactoryPSet = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSeedsB.RegionFactoryPSet,
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent")
)

mixedTripletStepSeeds = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSeeds.clone()

#track candidates
import FastSimulation.Tracking.TrackCandidateProducer_cfi
mixedTripletStepTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
    src = cms.InputTag("mixedTripletStepSeeds"),
    MinNumberOfCrossedLayers = 3
    #hitMasks = cms.InputTag("mixedTripletStepMasks","hitMasks"),
)

# tracks
mixedTripletStepTracks = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepTracks.clone(
    TTRHBuilder = 'WithoutRefit',
    Fitter = 'KFFittingSmootherThird',
    Propagator = 'PropagatorWithMaterial'
)

# final selection
mixedTripletStepClassifier1 = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepClassifier1.clone()
mixedTripletStepClassifier1.vertices = "firstStepPrimaryVerticesBeforeMixing"
mixedTripletStepClassifier2 = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepClassifier2.clone()
mixedTripletStepClassifier2.vertices = "firstStepPrimaryVerticesBeforeMixing"

mixedTripletStep = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStep.clone()

# Final sequence 
MixedTripletStep =  cms.Sequence(mixedTripletStepMasks
                                 +mixedTripletStepSeedsA
                                 +mixedTripletStepSeedsB
                                 +mixedTripletStepSeeds
                                 +mixedTripletStepTrackCandidates
                                 +mixedTripletStepTracks
                                 +mixedTripletStepClassifier1*mixedTripletStepClassifier2
                                 +mixedTripletStep                                 
                             )
