import FWCore.ParameterSet.Config as cms

#
# Very large impact parameter tracking: Iteration 5 using TOB + TEC ring 5 seeding
#

#HIT REMOVAL
largeD0step5Clusters = cms.EDFilter("TrackClusterRemover",
# To run this step, eliminating hits from all previous iterations ...   
#    trajectories = cms.InputTag("largeD0step4"),
#    oldClusterRemovalInfo = cms.InputTag("largeD0step4Clusters"),
#    pixelClusters = cms.InputTag("largeD0step4Clusters"),
#    stripClusters = cms.InputTag("largeD0step4Clusters"),

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
largeD0step5PixelRecHits = RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi.siPixelRecHits.clone()
largeD0step5StripRecHits = RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi.siStripMatchedRecHits.clone()
largeD0step5PixelRecHits.src = 'largeD0step5Clusters'
largeD0step5StripRecHits.ClusterProducer = 'largeD0step5Clusters'

#SEEDING LAYERS
import RecoTracker.TkSeedingLayers.TobTecLayerPairs_cfi
largeD0step5layerpairs = RecoTracker.TkSeedingLayers.TobTecLayerPairs_cfi.tobteclayerpairs.clone()
largeD0step5layerpairs.ComponentName = 'largeD0step5LayerPairs'
largeD0step5layerpairs.TOB.matchedRecHits = 'largeD0step5StripRecHits:matchedRecHit'
largeD0step5layerpairs.TEC.matchedRecHits = 'largeD0step5StripRecHits:matchedRecHit'

#SEEDS
from RecoTracker.TkSeedGenerator.GlobalPixelLessSeeds_cff import *
import RecoTracker.TkSeedGenerator.GlobalPixelLessSeeds_cfi
largeD0step5Seeds = RecoTracker.TkSeedGenerator.GlobalPixelLessSeeds_cfi.globalPixelLessSeeds.clone()
largeD0step5Seeds.OrderedHitsFactoryPSet.SeedingLayers = 'largeD0step5LayerPairs'
largeD0step5Seeds.RegionFactoryPSet.RegionPSet.ptMin = 0.9
largeD0step5Seeds.RegionFactoryPSet.RegionPSet.originRadius = 20.0
largeD0step5Seeds.RegionFactoryPSet.RegionPSet.originHalfLength = 30.0
#largeD0step5Seeds.propagator = cms.string('PropagatorWithMaterialPtMin09')

#TRAJECTORY MEASUREMENT
import RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi
largeD0step5MeasurementTracker = RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi.MeasurementTracker.clone()
largeD0step5MeasurementTracker.ComponentName = 'largeD0step5MeasurementTracker'
largeD0step5MeasurementTracker.pixelClusterProducer = 'largeD0step5Clusters'
largeD0step5MeasurementTracker.stripClusterProducer = 'largeD0step5Clusters'

#TRAJECTORY FILTERS (for inwards and outwards track building steps)
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi

largeD0step5CkfTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone()
largeD0step5CkfTrajectoryFilter.ComponentName = 'largeD0step5CkfTrajectoryFilter'
#largeD0step5CkfTrajectoryFilter.filterPset.maxLostHits = 1
#largeD0step5CkfTrajectoryFilter.filterPset.maxConsecLostHits = 2
largeD0step5CkfTrajectoryFilter.filterPset.minimumNumberOfHits = 6
largeD0step5CkfTrajectoryFilter.filterPset.minPt = 0.9
largeD0step5CkfTrajectoryFilter.filterPset.minHitsMinPt = 3

largeD0step5CkfInOutTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone()
largeD0step5CkfInOutTrajectoryFilter.ComponentName = 'largeD0step5CkfInOutTrajectoryFilter'
#largeD0step5CkfInOutTrajectoryFilter.filterPset.maxLostHits = 1
#largeD0step5CkfInOutTrajectoryFilter.filterPset.maxConsecLostHits = 2
largeD0step5CkfInOutTrajectoryFilter.filterPset.minimumNumberOfHits = 4
largeD0step5CkfInOutTrajectoryFilter.filterPset.minPt = 0.9
largeD0step5CkfInOutTrajectoryFilter.filterPset.minHitsMinPt = 3

#TRAJECTORY BUILDER
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
largeD0step5CkfTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone()
largeD0step5CkfTrajectoryBuilder.ComponentName = 'largeD0step5CkfTrajectoryBuilder'
largeD0step5CkfTrajectoryBuilder.MeasurementTrackerName = 'largeD0step5MeasurementTracker'
largeD0step5CkfTrajectoryBuilder.trajectoryFilterName = 'largeD0step5CkfTrajectoryFilter'
largeD0step5CkfTrajectoryBuilder.inOutTrajectoryFilterName = 'largeD0step5CkfInOutTrajectoryFilter'
largeD0step5CkfTrajectoryBuilder.useSameTrajFilter = False
largeD0step5CkfTrajectoryBuilder.minNrOfHitsForRebuild = 4
#largeD0step5CkfTrajectoryBuilder.maxCand = 5
#largeD0step5CkfTrajectoryBuilder.lostHitPenalty = 100.
#largeD0step5CkfTrajectoryBuilder.alwaysUseInvalidHits = False
#largeD0step5CkfTrajectoryBuilder.propagatorAlong = cms.string('PropagatorWithMaterialPtMin09')
#largeD0step5CkfTrajectoryBuilder.propagatorOpposite = cms.string('PropagatorWithMaterialOppositePtMin09')

#TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
largeD0step5TrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone()
largeD0step5TrackCandidates.SeedProducer = 'largeD0step5Seeds'
largeD0step5TrackCandidates.TrajectoryBuilder = 'largeD0step5CkfTrajectoryBuilder'
largeD0step5TrackCandidates.doSeedingRegionRebuilding = True
largeD0step5TrackCandidates.useHitsSplitting = True
largeD0step5TrackCandidates.cleanTrajectoryAfterInOut = False

#
# TRACK FITTING AND SMOOTHING
#

import TrackingTools.TrackFitters.RungeKuttaKFFittingSmootherESProducer_cfi
largeD0step5FittingSmootherWithOutlierRejection = TrackingTools.TrackFitters.RungeKuttaKFFittingSmootherESProducer_cfi.RKFittingSmoother.clone()
largeD0step5FittingSmootherWithOutlierRejection.ComponentName = 'largeD0step5FittingSmootherWithOutlierRejection'
largeD0step5FittingSmootherWithOutlierRejection.EstimateCut = 20
largeD0step5FittingSmootherWithOutlierRejection.MinNumberOfHits = 6
largeD0step5FittingSmootherWithOutlierRejection.Fitter = cms.string('largeD0step5RKFitter')
largeD0step5FittingSmootherWithOutlierRejection.Smoother = cms.string('largeD0step5RKSmoother')

# Also necessary to specify minimum number of hits after final track fit
import TrackingTools.TrackFitters.RungeKuttaKFTrajectoryFitterESProducer_cfi
import TrackingTools.TrackFitters.RungeKuttaKFTrajectorySmootherESProducer_cfi
largeD0step5RKTrajectoryFitter = TrackingTools.TrackFitters.RungeKuttaKFTrajectoryFitterESProducer_cfi.RKTrajectoryFitter.clone()
largeD0step5RKTrajectorySmoother = TrackingTools.TrackFitters.RungeKuttaKFTrajectorySmootherESProducer_cfi.RKTrajectorySmoother.clone()
largeD0step5RKTrajectoryFitter.ComponentName = cms.string('largeD0step5RKFitter')
largeD0step5RKTrajectorySmoother.ComponentName = cms.string('largeD0step5RKSmoother')
largeD0step5RKTrajectoryFitter.minHits = 6
largeD0step5RKTrajectorySmoother.minHits = 6

#TRACKS
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
largeD0step5WithMaterialTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
largeD0step5WithMaterialTracks.src = 'largeD0step5TrackCandidates'
largeD0step5WithMaterialTracks.clusterRemovalInfo = 'largeD0step5Clusters'
largeD0step5WithMaterialTracks.AlgorithmName = cms.string('iter5LargeD0')
largeD0step5WithMaterialTracks.Fitter = 'largeD0step5FittingSmootherWithOutlierRejection'

largeD0step5 = cms.Sequence(largeD0step5Clusters*
                          largeD0step5PixelRecHits*largeD0step5StripRecHits*
                          largeD0step5Seeds*
                          largeD0step5TrackCandidates*
                          largeD0step5WithMaterialTracks)
                          







