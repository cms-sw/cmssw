import FWCore.ParameterSet.Config as cms

# import the full tracking equivalent of this file
import RecoTracker.IterativeTracking.DetachedTripletStep_cff as _detachedTripletStep

# fast tracking mask producer
import FastSimulation.Tracking.FastTrackingMaskProducer_cfi
detachedTripletStepMasks = FastSimulation.Tracking.FastTrackingMaskProducer_cfi.fastTrackingMaskProducer.clone(
    trackCollection = _detachedTripletStep.detachedTripletStepClusters.trajectories,
    TrackQuality = _detachedTripletStep.detachedTripletStepClusters.TrackQuality,
    overrideTrkQuals = cms.InputTag('initialStep',"QualityMasks")
)

# trajectory seeds
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
detachedTripletStepSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    simTrackSelection = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.simTrackSelection.clone(
        pTMin = 0.02,
        maxD0 = 30.0,
        maxZ0 = 50
        ),
    minLayersCrossed = 3,
    #hitMasks = cms.InputTag("detachedTripletStepMasks","hitMasks"),
    hitCombinationMasks = cms.InputTag("detachedTripletStepMasks","hitCombinationMasks"),
    ptMin = _detachedTripletStep.detachedTripletStepSeeds.RegionFactoryPSet.RegionPSet.ptMin,
    originHalfLength = _detachedTripletStep.detachedTripletStepSeeds.RegionFactoryPSet.RegionPSet.originHalfLength,
    originRadius = _detachedTripletStep.detachedTripletStepSeeds.RegionFactoryPSet.RegionPSet.originRadius,
    layerList = _detachedTripletStep.detachedTripletStepSeedLayers.layerList.value()
    )

# track candidates
import FastSimulation.Tracking.TrackCandidateProducer_cfi
detachedTripletStepTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
    src = cms.InputTag("detachedTripletStepSeeds"),
    MinNumberOfCrossedLayers = 3
    #hitMasks = cms.InputTag("detachedTripletStepMasks","hitMasks"),
    )

# tracks 
detachedTripletStepTracks = _detachedTripletStep.detachedTripletStepTracks.clone(
    Fitter = 'KFFittingSmootherSecond',
    TTRHBuilder = 'WithoutRefit',
    Propagator = 'PropagatorWithMaterial'

)

#final selection
detachedTripletStepClassifier1 = _detachedTripletStep.detachedTripletStepClassifier1.clone()
detachedTripletStepClassifier1.vertices = "firstStepPrimaryVerticesBeforeMixing"
detachedTripletStepClassifier2 = _detachedTripletStep.detachedTripletStepClassifier2.clone()
detachedTripletStepClassifier2.vertices = "firstStepPrimaryVerticesBeforeMixing"
detachedTripletStep = _detachedTripletStep.detachedTripletStep.clone() 

# Final sequence 
DetachedTripletStep = cms.Sequence(detachedTripletStepMasks
                                   +detachedTripletStepSeeds
                                   +detachedTripletStepTrackCandidates
                                   +detachedTripletStepTracks
                                   +detachedTripletStepClassifier1*detachedTripletStepClassifier2
                                   +detachedTripletStep
                                   )
