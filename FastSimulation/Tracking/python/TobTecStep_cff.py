import FWCore.ParameterSet.Config as cms

# import the full tracking equivalent of this file
import RecoTracker.IterativeTracking.TobTecStep_cff
                                                                                                                                                                                                                                                              
# fast tracking mask producer 
from FastSimulation.Tracking.FastTrackingMaskProducer_cfi import fastTrackingMaskProducer as _fastTrackingMaskProducer                                                   
tobTecStepFastTrackingMasks = _fastTrackingMaskProducer.clone(                                                                                                                                                                                                 
    trackCollection = cms.InputTag("pixelLessStepTracks"),
    TrackQuality = RecoTracker.IterativeTracking.TobTecStep_cff.tobTecStepClusters.TrackQuality,
    overrideTrkQuals = cms.InputTag('pixelLessStep'),  
)

# trajectory seeds 
#triplet seeds
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
tobTecStepSeedsTripl = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    simTrackSelection = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.simTrackSelection.clone(
        pTMin = 0.3,
        maxD0 = -1,
        maxZ0 = -1
    ),
    minLayersCrossed = 4,
    hitMasks = cms.InputTag("tobTecStepFastTrackingMasks","hitMasks"),
    hitCombinationMasks = cms.InputTag("tobTecStepFastTrackingMasks","hitCombinationMasks"),
    ptMin = RecoTracker.IterativeTracking.TobTecStep_cff.tobTecStepSeedsTripl.RegionFactoryPSet.RegionPSet.ptMin,
    originHalfLength = RecoTracker.IterativeTracking.TobTecStep_cff.tobTecStepSeedsTripl.RegionFactoryPSet.RegionPSet.originHalfLength,
    originRadius = RecoTracker.IterativeTracking.TobTecStep_cff.tobTecStepSeedsTripl.RegionFactoryPSet.RegionPSet.originRadius,
    layerList = RecoTracker.IterativeTracking.TobTecStep_cff.tobTecStepSeedLayersTripl.layerList.value()
)
#pair seeds
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
tobTecStepSeedsPair = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    simTrackSelection = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.simTrackSelection.clone(
        pTMin = 0.3,
        maxD0 = 99.0,
        maxZ0 = 99
    ),
    minLayersCrossed = 4,
    hitMasks = cms.InputTag("tobTecStepFastTrackingMasks","hitMasks"),
    hitCombinationMasks = cms.InputTag("tobTecStepFastTrackingMasks","hitCombinationMasks"),
    ptMin = RecoTracker.IterativeTracking.TobTecStep_cff.tobTecStepSeedsPair.RegionFactoryPSet.RegionPSet.ptMin,
    originHalfLength = RecoTracker.IterativeTracking.TobTecStep_cff.tobTecStepSeedsPair.RegionFactoryPSet.RegionPSet.originHalfLength,
    originRadius = RecoTracker.IterativeTracking.TobTecStep_cff.tobTecStepSeedsPair.RegionFactoryPSet.RegionPSet.originRadius,
    layerList = RecoTracker.IterativeTracking.TobTecStep_cff.tobTecStepSeedLayersPair.layerList.value()
)
#
tobTecStepSeeds = RecoTracker.IterativeTracking.TobTecStep_cff.tobTecStepSeeds.clone()

# track candidate
import FastSimulation.Tracking.TrackCandidateProducer_cfi
tobTecStepTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
    src = cms.InputTag("tobTecStepSeeds"),
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
TobTecStep = cms.Sequence(tobTecStepFastTrackingMasks
                          +tobTecStepSeedsTripl
                          +tobTecStepSeedsPair
                          +tobTecStepSeeds
                          +tobTecStepTrackCandidates
                          +tobTecStepTracks
                          +tobTecStepSelector                          
                      )
