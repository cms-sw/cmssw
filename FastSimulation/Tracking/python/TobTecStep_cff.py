import FWCore.ParameterSet.Config as cms

# import the full tracking equivalent of this file
import RecoTracker.IterativeTracking.TobTecStep_cff
 
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
    minLayersCrossed = 4,
    layerList = RecoTracker.IterativeTracking.TobTecStep_cff.tobTecStepSeedLayersTripl.layerList.value(),
    RegionFactoryPSet = RecoTracker.IterativeTracking.TobTecStep_cff.tobTecStepSeedsTripl.RegionFactoryPSet,
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    )
#pair seeds
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
tobTecStepSeedsPair = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    minLayersCrossed = 4,
    layerList = RecoTracker.IterativeTracking.TobTecStep_cff.tobTecStepSeedLayersPair.layerList.value(),
    RegionFactoryPSet = RecoTracker.IterativeTracking.TobTecStep_cff.tobTecStepSeedsPair.RegionFactoryPSet,
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
    )
#
tobTecStepSeeds = RecoTracker.IterativeTracking.TobTecStep_cff.tobTecStepSeeds.clone()

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
                               +tobTecStepClassifier1*tobTecStepClassifier2                           +tobTecStep
                       )
