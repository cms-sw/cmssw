import FWCore.ParameterSet.Config as cms

# import the full tracking equivalent of this file
import RecoTracker.IterativeTracking.MixedTripletStep_cff

# simtrack id producer                                                                                                                                                         
import FastSimulation.Tracking.SimTrackIdProducer_cfi
mixedTripletStepSimTrackIds=FastSimulation.Tracking.SimTrackIdProducer_cfi.simTrackIdProducer.clone(
    trackCollection = cms.InputTag("pixelPairStepTracks"),
    TrackQuality = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepClusters.TrackQuality,
    maxChi2 = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepClusters.maxChi2,
    overrideTrkQuals = cms.InputTag('pixelPairStepSelector','pixelPairStep')
)

# trajectory seeds
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
mixedTripletStepSeedsA = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    simTrackSelection = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.simTrackSelection.clone(
        skipSimTrackIds = [
            cms.InputTag("detachedTripletStepSimTrackIds"),
            cms.InputTag("lowPtTripletStepSimTrackIds"),
            cms.InputTag("pixelPairStepSimTrackIds"),
            cms.InputTag("mixedTripletStepSimTrackIds")],
#        pTMin = 0.15,
 #       maxD0 = 10.0,
  #      maxZ0 = 30
        pTMin = 0,
        maxD0 = -1,
        maxZ0 = -1,
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
        skipSimTrackIds = [
            cms.InputTag("detachedTripletStepSimTrackIds"),
            cms.InputTag("lowPtTripletStepSimTrackIds"),
            cms.InputTag("pixelPairStepSimTrackIds"),
            cms.InputTag("mixedTripletStepSimTrackIds")],
#        pTMin = 0.15,
 #       maxD0 = 10.0,
  #      maxZ0 = 30
        pTMin = 0,
        maxD0 = -1,
        maxZ0 = -1,
        ),
    minLayersCrossed = 3,
    layerList = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSeedLayersB.layerList.value(),
    RegionFactoryPSet = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSeedsB.RegionFactoryPSet,
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
)

mixedTripletStepSeeds = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSeeds.clone()

#track candidates
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

# final selection
mixedTripletStepSelector = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSelector.clone()
mixedTripletStepSelector.vertices = "firstStepPrimaryVerticesBeforeMixing"
mixedTripletStep = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStep.clone()

# Final sequence 
MixedTripletStep =  cms.Sequence(mixedTripletStepSimTrackIds
                                 +mixedTripletStepSeedsA
                                 +mixedTripletStepSeedsB
                                 +mixedTripletStepSeeds
                                 +mixedTripletStepTrackCandidates
                                 +mixedTripletStepTracks
                                 +mixedTripletStepSelector
                                 +mixedTripletStep                                 
                                 )
