import FWCore.ParameterSet.Config as cms

# import the full tracking equivalent of this file
import RecoTracker.IterativeTracking.TobTecStep_cff as _standard
from FastSimulation.Tracking.SeedingMigration import _hitSetProducerToFactoryPSet

# fast tracking mask producer
import FastSimulation.Tracking.FastTrackerRecHitMaskProducer_cfi
tobTecStepMasks = FastSimulation.Tracking.FastTrackerRecHitMaskProducer_cfi.maskProducerFromClusterRemover(_standard.tobTecStepClusters)

# tracking regions
tobTecStepTrackingRegionsTripl = _standard.tobTecStepTrackingRegionsTripl.clone()

# trajectory seeds 
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
tobTecStepSeedsTripl = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    layerList = _standard.tobTecStepSeedLayersTripl.layerList.value(),
    trackingRegions = "tobTecStepTrackingRegionsTripl",
    hitMasks = cms.InputTag("tobTecStepMasks"),
)
tobTecStepSeedsTripl.seedFinderSelector.MultiHitGeneratorFactory = _hitSetProducerToFactoryPSet(_standard.tobTecStepHitTripletsTripl)
tobTecStepSeedsTripl.seedFinderSelector.MultiHitGeneratorFactory.SeedComparitorPSet=cms.PSet(  ComponentName = cms.string( "none" ) )
tobTecStepSeedsTripl.seedFinderSelector.MultiHitGeneratorFactory.refitHits = False

# pair tracking regions
tobTecStepTrackingRegionsPair = _standard.tobTecStepTrackingRegionsPair.clone()

#pair seeds
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
tobTecStepSeedsPair = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    layerList = _standard.tobTecStepSeedLayersPair.layerList.value(),
    trackingRegions = "tobTecStepTrackingRegionsPair",
    hitMasks = cms.InputTag("tobTecStepMasks"),
)

#
tobTecStepSeeds = _standard.tobTecStepSeeds.clone()

# track candidate
import FastSimulation.Tracking.TrackCandidateProducer_cfi
tobTecStepTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
    MinNumberOfCrossedLayers = 3,
    src = cms.InputTag("tobTecStepSeeds"),
    hitMasks = cms.InputTag("tobTecStepMasks"),
)

# track fitters
from RecoTracker.IterativeTracking.TobTecStep_cff import tobTecFlexibleKFFittingSmoother,tobTecStepRKTrajectorySmootherForLoopers,tobTecStepRKTrajectorySmoother,tobTecStepRKTrajectoryFitterForLoopers,tobTecStepRKTrajectoryFitter,tobTecStepFitterSmootherForLoopers,tobTecStepFitterSmoother

# tracks 
tobTecStepTracks = _standard.tobTecStepTracks.clone(TTRHBuilder = 'WithoutRefit')

# final selection
tobTecStepClassifier1 = _standard.tobTecStepClassifier1.clone()
tobTecStepClassifier1.vertices = "firstStepPrimaryVerticesBeforeMixing"
tobTecStepClassifier2 = _standard.tobTecStepClassifier2.clone()
tobTecStepClassifier2.vertices = "firstStepPrimaryVerticesBeforeMixing"

tobTecStep = _standard.tobTecStep.clone()



# Final sequence 
TobTecStep = cms.Sequence(tobTecStepMasks
                          +tobTecStepTrackingRegionsTripl
                          +tobTecStepSeedsTripl
                          +tobTecStepTrackingRegionsPair
                           +tobTecStepSeedsPair
                           +tobTecStepSeeds
                           +tobTecStepTrackCandidates
                           +tobTecStepTracks
                          +tobTecStepClassifier1*tobTecStepClassifier2                           +tobTecStep
                       )
