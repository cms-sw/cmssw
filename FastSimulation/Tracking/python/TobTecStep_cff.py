import FWCore.ParameterSet.Config as cms

# import the full tracking equivalent of this file
import RecoTracker.IterativeTracking.TobTecStep_cff
                                                                                                                                                                                                                                                              
# fast tracking mask producer 
from FastSimulation.Tracking.FastTrackingMaskProducer_cfi import fastTrackingMaskProducer as _fastTrackingMaskProducer                                                   
tobTecStepMasks = _fastTrackingMaskProducer.clone(                                                                                                                                                                                                 
    trackCollection = cms.InputTag("pixelLessStepTracks"),
    TrackQuality = RecoTracker.IterativeTracking.TobTecStep_cff.tobTecStepClusters.TrackQuality,
    overrideTrkQuals = cms.InputTag('pixelLessStep',"QualityMasks"),  
    oldHitCombinationMasks = cms.InputTag("pixelLessStepMasks","hitCombinationMasks"),
    oldHitMasks = cms.InputTag("pixelLessStepMasks","hitMasks")
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
    #hitMasks = cms.InputTag("tobTecStepMasks","hitMasks"),
    hitCombinationMasks = cms.InputTag("tobTecStepMasks","hitCombinationMasks"),
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
    #hitMasks = cms.InputTag("tobTecStepMasks","hitMasks"),
    hitCombinationMasks = cms.InputTag("tobTecStepMasks","hitCombinationMasks"),
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
    #hitMasks = cms.InputTag("tobTecStepMasks","hitMasks"),
)

# tracks 
tobTecStepTracks = RecoTracker.IterativeTracking.TobTecStep_cff.tobTecStepTracks.clone(
    TTRHBuilder = 'WithoutRefit',
    Fitter = 'KFFittingSmootherFifth',
    Propagator = 'PropagatorWithMaterial'
)

# final selection
tobTecStepClassifier1 = RecoTracker.IterativeTracking.TobTecStep_cff.tobTecStepClassifier1.clone()
tobTecStepClassifier1.vertices = "firstStepPrimaryVerticesBeforeMixing"
tobTecStepClassifier2 = RecoTracker.IterativeTracking.TobTecStep_cff.tobTecStepClassifier2.clone()
tobTecStepClassifier2.vertices = "firstStepPrimaryVerticesBeforeMixing"

tobTecStep = RecoTracker.IterativeTracking.TobTecStep_cff.tobTecStep.clone()



# Final sequence 
TobTecStep = cms.Sequence(tobTecStepMasks
                          +tobTecStepSeedsTripl
                          +tobTecStepSeedsPair
                          +tobTecStepSeeds
                          +tobTecStepTrackCandidates
                          +tobTecStepTracks
                          +tobTecStepClassifier1*tobTecStepClassifier2
                          +tobTecStep
                      )
