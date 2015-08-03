
import FWCore.ParameterSet.Config as cms

# import the full tracking equivalent of this file
import RecoTracker.IterativeTracking.LowPtTripletStep_cff

# fast tracking mask producer
from FastSimulation.Tracking.FastTrackingMaskProducer_cfi import fastTrackingMaskProducer as _fastTrackingMaskProducer 
lowPtTripletStepMasks = _fastTrackingMaskProducer.clone(
    trackCollection = cms.InputTag("detachedTripletStepTracks"),
    TrackQuality = RecoTracker.IterativeTracking.LowPtTripletStep_cff.lowPtTripletStepClusters.TrackQuality,
    overrideTrkQuals = cms.InputTag('detachedTripletStep',"QualityMasks"),                        
    oldHitCombinationMasks = cms.InputTag("detachedTripletStepMasks","hitCombinationMasks"),
    oldHitMasks = cms.InputTag("detachedTripletStepMasks","hitMasks")
    )


# trajectory seeds
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
lowPtTripletStepSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    simTrackSelection = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.simTrackSelection.clone(
        pTMin = 0.1,
        maxD0 = 5.0,
        maxZ0 = 50
    ),
    minLayersCrossed = 3,
    #hitMasks = cms.InputTag("lowPtTripletStepMasks","hitMasks"),
    hitCombinationMasks = cms.InputTag("lowPtTripletStepMasks","hitCombinationMasks"),
    nSigmaZ = RecoTracker.IterativeTracking.LowPtTripletStep_cff.lowPtTripletStepSeeds.RegionFactoryPSet.RegionPSet.nSigmaZ,
    ptMin = RecoTracker.IterativeTracking.LowPtTripletStep_cff.lowPtTripletStepSeeds.RegionFactoryPSet.RegionPSet.ptMin,
    originRadius = RecoTracker.IterativeTracking.LowPtTripletStep_cff.lowPtTripletStepSeeds.RegionFactoryPSet.RegionPSet.originRadius,
    layerList = RecoTracker.IterativeTracking.LowPtTripletStep_cff.lowPtTripletStepSeedLayers.layerList.value()
)

# track candidates
import FastSimulation.Tracking.TrackCandidateProducer_cfi
lowPtTripletStepTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
    src = cms.InputTag("lowPtTripletStepSeeds"),
    MinNumberOfCrossedLayers = 3
    #hitMasks = cms.InputTag("lowPtTripletStepMasks","hitMasks"),
)

# tracks
lowPtTripletStepTracks = RecoTracker.IterativeTracking.LowPtTripletStep_cff.lowPtTripletStepTracks.clone(
    Fitter = 'KFFittingSmootherSecond',
    TTRHBuilder = 'WithoutRefit',
    Propagator = 'PropagatorWithMaterial'
)

# final selection
lowPtTripletStep = RecoTracker.IterativeTracking.LowPtTripletStep_cff.lowPtTripletStep.clone()
lowPtTripletStep.vertices = "firstStepPrimaryVerticesBeforeMixing"

# Final swquence 
LowPtTripletStep = cms.Sequence(lowPtTripletStepMasks
                                +lowPtTripletStepSeeds
                                +lowPtTripletStepTrackCandidates
                                +lowPtTripletStepTracks  
                                +lowPtTripletStep   
                                )
