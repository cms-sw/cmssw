import FWCore.ParameterSet.Config as cms

#
# Very large impact parameter tracking: Iteration 2 using pixel-pair seeding
#

#HIT REMOVAL
largeD0step2Clusters = cms.EDFilter("TrackClusterRemover",

# To run this step, eliminating hits from all previous iterations ...   
#    trajectories = cms.InputTag("largeD0step1"),
#    oldClusterRemovalInfo = cms.InputTag("largeD0step1Clusters"),
#    pixelClusters = cms.InputTag("largeD0step1Clusters"),
#    stripClusters = cms.InputTag("largeD0step1Clusters"),

# To run this step independently of the other large d0 tracking iterations ...
    trajectories = cms.InputTag("pixellessStep"),
    oldClusterRemovalInfo = cms.InputTag("fourthClusters"),
    pixelClusters = cms.InputTag("fourthClusters"),
    stripClusters = cms.InputTag("fourthClusters"),

# To run it independently of all tracking iterations ...
#    trajectories = cms.InputTag("zeroStepFilter"),
#    pixelClusters = cms.InputTag("siPixelClusters"),
#    stripClusters = cms.InputTag("siStripClusters"),
                                  
    Common = cms.PSet(
       maxChi2 = cms.double(30.0)
# To run it independently of all tracking iterations, also need ...
#       maxChi2 = cms.double(0.0)
    )
)

# Propagator taking into account momentum uncertainty in multiple
# scattering calculation.
from TrackingTools.MaterialEffects.Propagators_PtMin09_cff import *

#TRACKER HITS
import RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi
import RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi
largeD0step2PixelRecHits = RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi.siPixelRecHits.clone()
largeD0step2StripRecHits = RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi.siStripMatchedRecHits.clone()
largeD0step2PixelRecHits.src = 'largeD0step2Clusters'
largeD0step2StripRecHits.ClusterProducer = 'largeD0step2Clusters'

#SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelLayerPairs_cfi
largeD0step2layerpairs = RecoTracker.TkSeedingLayers.PixelLayerPairs_cfi.pixellayerpairs.clone()
largeD0step2layerpairs.ComponentName = 'largeD0step2LayerPairs'
largeD0step2layerpairs.BPix.HitProducer = 'largeD0step2PixelRecHits'
largeD0step2layerpairs.FPix.HitProducer = 'largeD0step2PixelRecHits'

#SEEDS
import RecoTracker.TkSeedGenerator.GlobalPixelSeeds_cfi
largeD0step2Seeds = RecoTracker.TkSeedGenerator.GlobalPixelSeeds_cfi.globalPixelSeeds.clone()
largeD0step2Seeds.OrderedHitsFactoryPSet.SeedingLayers = 'largeD0step2LayerPairs'
largeD0step2Seeds.RegionFactoryPSet.RegionPSet.ptMin = 0.6
largeD0step2Seeds.RegionFactoryPSet.RegionPSet.originRadius = 2.5
largeD0step2Seeds.RegionFactoryPSet.RegionPSet.originHalfLength = 15
largeD0step2Seeds.propagator = cms.string('PropagatorWithMaterialPtMin09')
# The fast-helix fit doesn't work well for large d0 pixel pair seeding.
largeD0step2Seeds.UseFastHelix = False

#TRAJECTORY MEASUREMENT
import RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi
largeD0step2MeasurementTracker = RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi.MeasurementTracker.clone()
largeD0step2MeasurementTracker.ComponentName = 'largeD0step2MeasurementTracker'
largeD0step2MeasurementTracker.pixelClusterProducer = 'largeD0step2Clusters'
largeD0step2MeasurementTracker.stripClusterProducer = 'largeD0step2Clusters'

#TRAJECTORY FILTERS (for inwards and outwards track building steps)
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi

largeD0step2CkfTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone()
largeD0step2CkfTrajectoryFilter.ComponentName = 'largeD0step2CkfTrajectoryFilter'
#largeD0step2CkfTrajectoryFilter.filterPset.maxLostHits = 1
#largeD0step2CkfTrajectoryFilter.filterPset.maxConstep2LostHits = 2
largeD0step2CkfTrajectoryFilter.filterPset.minimumNumberOfHits = 6
largeD0step2CkfTrajectoryFilter.filterPset.minPt = 0.6
largeD0step2CkfTrajectoryFilter.filterPset.minHitsMinPt = 3

#TRAJECTORY BUILDER
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
largeD0step2CkfTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone()
largeD0step2CkfTrajectoryBuilder.ComponentName = 'largeD0step2CkfTrajectoryBuilder'
largeD0step2CkfTrajectoryBuilder.MeasurementTrackerName = 'largeD0step2MeasurementTracker'
largeD0step2CkfTrajectoryBuilder.trajectoryFilterName = 'largeD0step2CkfTrajectoryFilter'
largeD0step2CkfTrajectoryBuilder.useSameTrajFilter = True
largeD0step2CkfTrajectoryBuilder.minNrOfHitsForRebuild = 6
#largeD0step2CkfTrajectoryBuilder.maxCand = 5
#largeD0step2CkfTrajectoryBuilder.lostHitPenalty = 100.
#largeD0step2CkfTrajectoryBuilder.alwaysUseInvalidHits = False
largeD0step2CkfTrajectoryBuilder.propagatorAlong = cms.string('PropagatorWithMaterialPtMin09')
largeD0step2CkfTrajectoryBuilder.propagatorOpposite = cms.string('PropagatorWithMaterialOppositePtMin09')


#TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
largeD0step2TrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone()
largeD0step2TrackCandidates.SeedProducer = 'largeD0step2Seeds'
largeD0step2TrackCandidates.TrajectoryBuilder = 'largeD0step2CkfTrajectoryBuilder'
largeD0step2TrackCandidates.doSeedingRegionRebuilding = True
largeD0step2TrackCandidates.useHitsSplitting = True
largeD0step2TrackCandidates.cleanTrajectoryAfterInOut = True

#
# TRACK FITTING AND SMOOTHING
#

import TrackingTools.TrackFitters.RungeKuttaKFFittingSmootherESProducer_cfi
largeD0step2FittingSmootherWithOutlierRejection = TrackingTools.TrackFitters.RungeKuttaKFFittingSmootherESProducer_cfi.RKFittingSmoother.clone()
largeD0step2FittingSmootherWithOutlierRejection.ComponentName = 'largeD0step2FittingSmootherWithOutlierRejection'
largeD0step2FittingSmootherWithOutlierRejection.EstimateCut = 20
largeD0step2FittingSmootherWithOutlierRejection.MinNumberOfHits = 6
largeD0step2FittingSmootherWithOutlierRejection.Fitter = cms.string('largeD0step2RKFitter')
largeD0step2FittingSmootherWithOutlierRejection.Smoother = cms.string('largeD0step2RKSmoother')

# Also necessary to specify minimum number of hits after final track fit
import TrackingTools.TrackFitters.RungeKuttaKFTrajectoryFitterESProducer_cfi
import TrackingTools.TrackFitters.RungeKuttaKFTrajectorySmootherESProducer_cfi
largeD0step2RKTrajectoryFitter = TrackingTools.TrackFitters.RungeKuttaKFTrajectoryFitterESProducer_cfi.RKTrajectoryFitter.clone()
largeD0step2RKTrajectorySmoother = TrackingTools.TrackFitters.RungeKuttaKFTrajectorySmootherESProducer_cfi.RKTrajectorySmoother.clone()
largeD0step2RKTrajectoryFitter.ComponentName = cms.string('largeD0step2RKFitter')
largeD0step2RKTrajectorySmoother.ComponentName = cms.string('largeD0step2RKSmoother')
largeD0step2RKTrajectoryFitter.minHits = 6
largeD0step2RKTrajectorySmoother.minHits = 6

#TRACKS
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
largeD0step2WithMaterialTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
largeD0step2WithMaterialTracks.src = 'largeD0step2TrackCandidates'
largeD0step2WithMaterialTracks.clusterRemovalInfo = 'largeD0step2Clusters'
largeD0step2WithMaterialTracks.AlgorithmName = cms.string('iter2LargeD0')
largeD0step2WithMaterialTracks.Fitter = 'largeD0step2FittingSmootherWithOutlierRejection'

largeD0step2 = cms.Sequence(largeD0step2Clusters*
                            largeD0step2PixelRecHits*largeD0step2StripRecHits*
                            largeD0step2Seeds*
                            largeD0step2TrackCandidates*
                            largeD0step2WithMaterialTracks)
                          







