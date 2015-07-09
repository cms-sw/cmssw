import FWCore.ParameterSet.Config as cms

# import the full tracking equivalent of this file
import RecoTracker.IterativeTracking.DetachedTripletStep_cff
# simtrack id producer
import FastSimulation.Tracking.SimTrackIdProducer_cfi
detachedTripletStepSimTrackIds = FastSimulation.Tracking.SimTrackIdProducer_cfi.simTrackIdProducer.clone(
    #tracjectories = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepClusters.trajectories.value(),
    trackCollection = cms.InputTag("initialStepTracks"),
    TrackQuality = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepClusters.TrackQuality,
    maxChi2 = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepClusters.maxChi2,
    trackClassifier                          = cms.InputTag('initialStep',"QualityMasks")
    )
# trajectory seeds
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
detachedTripletStepSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    simTrackSelection = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.simTrackSelection.clone(
        skipSimTrackIds = [cms.InputTag("detachedTripletStepSimTrackIds")],
        pTMin = 0.02,
        maxD0 = 30.0,
        maxZ0 = 50
        ),
    minLayersCrossed = 3,
    ptMin = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepSeeds.RegionFactoryPSet.RegionPSet.ptMin,
    originHalfLength = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepSeeds.RegionFactoryPSet.RegionPSet.originHalfLength,
    originRadius = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepSeeds.RegionFactoryPSet.RegionPSet.originRadius,
    layerList = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepSeedLayers.layerList.value()
    )

# track candidates
import FastSimulation.Tracking.TrackCandidateProducer_cfi
detachedTripletStepTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
    src = cms.InputTag("detachedTripletStepSeeds"),
    MinNumberOfCrossedLayers = 3
    )

# tracks 
detachedTripletStepTracks = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepTracks.clone(
    Fitter = 'KFFittingSmootherSecond',
    TTRHBuilder = 'WithoutRefit',
    Propagator = 'PropagatorWithMaterial'

)

#final selection
detachedTripletStepClassifier1 = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepClassifier1.clone()
detachedTripletStepClassifier1.vertices = "firstStepPrimaryVerticesBeforeMixing"
detachedTripletStepClassifier2 = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepClassifier2.clone()
detachedTripletStepClassifier2.vertices = "firstStepPrimaryVerticesBeforeMixing"
detachedTripletStep = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStep.clone() 

# Final sequence 
DetachedTripletStep = cms.Sequence(detachedTripletStepSimTrackIds
                                   +detachedTripletStepSeeds
                                   +detachedTripletStepTrackCandidates
                                   +detachedTripletStepTracks
                                   +detachedTripletStepClassifier1*detachedTripletStepClassifier2
                                   +detachedTripletStep
                                   )
