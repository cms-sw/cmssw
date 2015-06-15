import FWCore.ParameterSet.Config as cms

# import the full tracking equivalent of this file
import RecoTracker.IterativeTracking.LowPtTripletStep_cff

# simtrack id producer                                                                
import FastSimulation.Tracking.SimTrackIdProducer_cfi
lowPtTripletStepSimTrackIds = FastSimulation.Tracking.SimTrackIdProducer_cfi.simTrackIdProducer.clone(
    trackCollection = cms.InputTag("detachedTripletStepTracks"),
    TrackQuality = RecoTracker.IterativeTracking.LowPtTripletStep_cff.lowPtTripletStepClusters.TrackQuality,
    maxChi2 = RecoTracker.IterativeTracking.LowPtTripletStep_cff.lowPtTripletStepClusters.maxChi2,
    overrideTrkQuals = cms.InputTag('detachedTripletStep')
)

# trajectory seeds
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
lowPtTripletStepSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    simTrackSelection = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.simTrackSelection.clone(
        skipSimTrackIds = [
            cms.InputTag("detachedTripletStepSimTrackIds"),
            cms.InputTag("lowPtTripletStepSimTrackIds")],
        pTMin = 0.1,
        maxD0 = 5.0,
        maxZ0 = 50
    ),
    minLayersCrossed = 3,
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
)

# tracks
lowPtTripletStepTracks = RecoTracker.IterativeTracking.LowPtTripletStep_cff.lowPtTripletStepTracks.clone(
    Fitter = 'KFFittingSmootherSecond',
    TTRHBuilder = 'WithoutRefit',
    Propagator = 'PropagatorWithMaterial'
)

# final selection
lowPtTripletStepSelector = RecoTracker.IterativeTracking.LowPtTripletStep_cff.lowPtTripletStepSelector.clone()
lowPtTripletStepSelector.vertices = "firstStepPrimaryVerticesBeforeMixing"

# Final swquence 
LowPtTripletStep = cms.Sequence(lowPtTripletStepSimTrackIds
                                +lowPtTripletStepSeeds
                                +lowPtTripletStepTrackCandidates
                                +lowPtTripletStepTracks  
                                +lowPtTripletStepSelector   
                                )
