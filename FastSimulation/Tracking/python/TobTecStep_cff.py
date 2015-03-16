import FWCore.ParameterSet.Config as cms

# import the full tracking equivalent of this file
import RecoTracker.IterativeTracking.TobTecStep_cff

# simtrack id producer                                                                                                                                                         
import FastSimulation.Tracking.SimTrackIdProducer_cfi
tobTecStepSimTrackIds = FastSimulation.Tracking.SimTrackIdProducer_cfi.simTrackIdProducer.clone(
    trajectories = cms.InputTag("pixelLessStepTracks"),
    TrackQuality = RecoTracker.IterativeTracking.TobTecStep_cff.tobTecStepClusters.TrackQuality,
    maxChi2 = RecoTracker.IterativeTracking.TobTecStep_cff.tobTecStepClusters.maxChi2,
)

# trajectory seeds 
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
tobTecStepSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    simTrackSelection = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.simTrackSelection.clone(
        skipSimTrackIds = [
            cms.InputTag("detachedTripletStepSimTrackIds"),
            cms.InputTag("lowPtTripletStepSimTrackIds"),
            cms.InputTag("pixelPairStepSimTrackIds"),
            cms.InputTag("mixedTripletStepSimTrackIds"),
            cms.InputTag("pixelLessStepSimTrackIds"),
            cms.InputTag("tobTecStepSimTrackIds")],
        pTMin = 0.3,
        maxD0 = 99.0,
        maxZ0 = 99
    ),
    minLayersCrossed = 4,
    ptMin = RecoTracker.IterativeTracking.TobTecStep_cff.tobTecStepSeedsPair.RegionFactoryPSet.RegionPSet.ptMin,
    originHalfLength = RecoTracker.IterativeTracking.TobTecStep_cff.tobTecStepSeedsPair.RegionFactoryPSet.RegionPSet.originHalfLength,
    originRadius = RecoTracker.IterativeTracking.TobTecStep_cff.tobTecStepSeedsPair.RegionFactoryPSet.RegionPSet.originRadius,
    layerList = RecoTracker.IterativeTracking.TobTecStep_cff.tobTecStepSeedLayersPair.layerList.value()
)

# track candidate
import FastSimulation.Tracking.TrackCandidateProducer_cfi
tobTecStepTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
    SeedProducer = cms.InputTag("tobTecStepSeeds"),
    MinNumberOfCrossedLayers = 3
)

# tracks 
tobTecStepTracks = RecoTracker.IterativeTracking.TobTecStep_cff.tobTecStepTracks.clone(
    TTRHBuilder = 'WithoutRefit',
    Fitter = 'KFFittingSmootherFifth',
    Propagator = 'PropagatorWithMaterial'
)

# final selection
tobTecStepSelector = RecoTracker.IterativeTracking.TobTecStep_cff.tobTecStepSelector.clone()
tobTecStepSelector.vertices = "firstStepPrimaryVerticesBeforeMixing"

# Final sequence 
TobTecStep = cms.Sequence(tobTecStepSimTrackIds
                          +tobTecStepSeeds
                          +tobTecStepTrackCandidates
                          +tobTecStepTracks
                          +tobTecStepSelector                          
                      )
