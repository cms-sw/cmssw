import FWCore.ParameterSet.Config as cms

# import the full tracking equivalent of this file
import RecoTracker.IterativeTracking.TobTecStep_cff as _standard

# fast tracking mask producer
import FastSimulation.Tracking.FastTrackerRecHitMaskProducer_cfi
tobTecStepMasks = FastSimulation.Tracking.FastTrackerRecHitMaskProducer_cfi.maskProducerFromClusterRemover(_standard.tobTecStepClusters)

# trajectory seeds 
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
tobTecStepSeedsTripl = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    layerList = _standard.tobTecStepSeedLayersTripl.layerList.value(),
    RegionFactoryPSet = _standard.tobTecStepSeedsTripl.RegionFactoryPSet,
    hitMasks = cms.InputTag("tobTecStepMasks"),
)
tobTecStepSeedsTripl.seedFinderSelector.MultiHitGeneratorFactory = _standard.tobTecStepSeedsTripl.OrderedHitsFactoryPSet.GeneratorPSet
tobTecStepSeedsTripl.seedFinderSelector.MultiHitGeneratorFactory.SeedComparitorPSet=cms.PSet(  ComponentName = cms.string( "none" ) )
tobTecStepSeedsTripl.seedFinderSelector.MultiHitGeneratorFactory.refitHits = False

#pair seeds
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
tobTecStepSeedsPair = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    layerList = _standard.tobTecStepSeedLayersPair.layerList.value(),
    RegionFactoryPSet = _standard.tobTecStepSeedsPair.RegionFactoryPSet,
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
                          +tobTecStepSeedsTripl
                           +tobTecStepSeedsPair
                           +tobTecStepSeeds
                           +tobTecStepTrackCandidates
                           +tobTecStepTracks
                          +tobTecStepClassifier1*tobTecStepClassifier2                           +tobTecStep
                       )
