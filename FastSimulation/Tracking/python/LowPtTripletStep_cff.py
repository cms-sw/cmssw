
import FWCore.ParameterSet.Config as cms

# import the full tracking equivalent of this file
import RecoTracker.IterativeTracking.LowPtTripletStep_cff

# fast tracking mask producer
import FastSimulation.Tracking.FastTrackerRecHitMaskProducer_cfi
lowPtTripletStepMasks = FastSimulation.Tracking.FastTrackerRecHitMaskProducer_cfi.maskProducerFromClusterRemover(RecoTracker.IterativeTracking.LowPtTripletStep_cff.lowPtTripletStepClusters)

# trajectory seeds
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
lowPtTripletStepSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    minLayersCrossed = 3,
    layerList = RecoTracker.IterativeTracking.InitialStep_cff.initialStepSeedLayers.layerList.value(),
    RegionFactoryPSet = RecoTracker.IterativeTracking.InitialStep_cff.initialStepSeeds.RegionFactoryPSet,
    hitMasks = cms.InputTag("lowPtTripletStepMasks"),
)

# track candidates
import FastSimulation.Tracking.TrackCandidateProducer_cfi
lowPtTripletStepTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
    src = cms.InputTag("lowPtTripletStepSeeds"),
    MinNumberOfCrossedLayers = 3,
    hitMasks = cms.InputTag("lowPtTripletStepMasks"),
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
