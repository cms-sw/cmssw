import FWCore.ParameterSet.Config as cms

#
# Very large impact parameter tracking: Iteration 1 using pixel-triplet seeding
#

#HIT REMOVAL
trkfilter1 = cms.EDFilter("QualityFilter",
    TrackQuality = cms.string('highPurity'),
    recTracks = cms.InputTag("tobtecStep")
)

largeD0step1Clusters = cms.EDFilter("TrackClusterRemover",
# To run this step, eliminating hits from all previous iterations ...   
#    trajectories = cms.InputTag("tobtecStep"),
    trajectories = cms.InputTag("trkfilter1"),
    oldClusterRemovalInfo = cms.InputTag("fifthClusters"),
    pixelClusters = cms.InputTag("fifthClusters"),
    stripClusters = cms.InputTag("fifthClusters"),

# To run it independently of all tracking iterations ...
#    trajectories = cms.InputTag("zeroStepFilter"),
#    pixelClusters = cms.InputTag("siPixelClusters"),
#    stripClusters = cms.InputTag("siStripClusters"),

    Common = cms.PSet(
       maxChi2 = cms.double(30.0)
# To run it independently of all tracking iterations, also need ...
#      maxChi2 = cms.double(0.0)
    )
)

# Propagator taking into account momentum uncertainty in multiple
# scattering calculation.
from TrackingTools.MaterialEffects.Propagators_PtMin09_cff import *

#TRACKER HITS
import RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi
import RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi
largeD0step1PixelRecHits = RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi.siPixelRecHits.clone()
largeD0step1StripRecHits = RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi.siStripMatchedRecHits.clone()
largeD0step1PixelRecHits.src = 'largeD0step1Clusters'
largeD0step1StripRecHits.ClusterProducer = 'largeD0step1Clusters'

#SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi
largeD0step1layertriplets = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.pixellayertriplets.clone()
largeD0step1layertriplets.ComponentName = 'largeD0step1LayerTriplets'
largeD0step1layertriplets.BPix.HitProducer = 'largeD0step1PixelRecHits'
largeD0step1layertriplets.FPix.HitProducer = 'largeD0step1PixelRecHits'

#SEEDS
import RecoTracker.TkSeedGenerator.GlobalSeedsFromTripletsWithVertices_cfi
largeD0step1Seeds = RecoTracker.TkSeedGenerator.GlobalSeedsFromTripletsWithVertices_cfi.globalSeedsFromTripletsWithVertices.clone()
largeD0step1Seeds.OrderedHitsFactoryPSet.SeedingLayers = 'largeD0step1LayerTriplets'
largeD0step1Seeds.RegionFactoryPSet.RegionPSet.ptMin = 0.9
largeD0step1Seeds.RegionFactoryPSet.RegionPSet.originRadius = 2.5
largeD0step1Seeds.RegionFactoryPSet.RegionPSet.originHalfLength = 15
#largeD0step1Seeds.propagator = cms.string('PropagatorWithMaterialPtMin09')

#TRAJECTORY MEASUREMENT
import RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi
largeD0step1MeasurementTracker = RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi.MeasurementTracker.clone()
largeD0step1MeasurementTracker.ComponentName = 'largeD0step1MeasurementTracker'
largeD0step1MeasurementTracker.pixelClusterProducer = 'largeD0step1Clusters'
largeD0step1MeasurementTracker.stripClusterProducer = 'largeD0step1Clusters'

#TRAJECTORY FILTERS (for inwards and outwards track building steps)
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi

largeD0step1CkfTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone()
largeD0step1CkfTrajectoryFilter.ComponentName = 'largeD0step1CkfTrajectoryFilter'
#largeD0step1CkfTrajectoryFilter.filterPset.maxLostHits = 1
#largeD0step1CkfTrajectoryFilter.filterPset.maxConsecLostHits = 2
largeD0step1CkfTrajectoryFilter.filterPset.minimumNumberOfHits = 6
largeD0step1CkfTrajectoryFilter.filterPset.minPt = 0.9
largeD0step1CkfTrajectoryFilter.filterPset.minHitsMinPt = 3

#TRAJECTORY BUILDER
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
largeD0step1CkfTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone()
largeD0step1CkfTrajectoryBuilder.ComponentName = 'largeD0step1CkfTrajectoryBuilder'
largeD0step1CkfTrajectoryBuilder.MeasurementTrackerName = 'largeD0step1MeasurementTracker'
largeD0step1CkfTrajectoryBuilder.trajectoryFilterName = 'largeD0step1CkfTrajectoryFilter'
largeD0step1CkfTrajectoryBuilder.useSameTrajFilter = True
largeD0step1CkfTrajectoryBuilder.minNrOfHitsForRebuild = 6
#largeD0step1CkfTrajectoryBuilder.maxCand = 5
#largeD0step1CkfTrajectoryBuilder.lostHitPenalty = 100.
#largeD0step1CkfTrajectoryBuilder.alwaysUseInvalidHits = False
#largeD0step1CkfTrajectoryBuilder.propagatorAlong = cms.string('PropagatorWithMaterialPtMin09')
#largeD0step1CkfTrajectoryBuilder.propagatorOpposite = cms.string('PropagatorWithMaterialOppositePtMin09')

#TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
largeD0step1TrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone()
largeD0step1TrackCandidates.SeedProducer = 'largeD0step1Seeds'
largeD0step1TrackCandidates.TrajectoryBuilder = 'largeD0step1CkfTrajectoryBuilder'
largeD0step1TrackCandidates.doSeedingRegionRebuilding = True
largeD0step1TrackCandidates.useHitsSplitting = True
largeD0step1TrackCandidates.cleanTrajectoryAfterInOut = True

#
# TRACK FITTING AND SMOOTHING
#

import TrackingTools.TrackFitters.RungeKuttaKFFittingSmootherESProducer_cfi
largeD0step1FittingSmootherWithOutlierRejection = TrackingTools.TrackFitters.RungeKuttaKFFittingSmootherESProducer_cfi.RKFittingSmoother.clone()
largeD0step1FittingSmootherWithOutlierRejection.ComponentName = 'largeD0step1FittingSmootherWithOutlierRejection'
largeD0step1FittingSmootherWithOutlierRejection.EstimateCut = 20
largeD0step1FittingSmootherWithOutlierRejection.MinNumberOfHits = 6
largeD0step1FittingSmootherWithOutlierRejection.Fitter = cms.string('largeD0step1RKFitter')
largeD0step1FittingSmootherWithOutlierRejection.Smoother = cms.string('largeD0step1RKSmoother')

# Also necessary to specify minimum number of hits after final track fit
import TrackingTools.TrackFitters.RungeKuttaKFTrajectoryFitterESProducer_cfi
import TrackingTools.TrackFitters.RungeKuttaKFTrajectorySmootherESProducer_cfi
largeD0step1RKTrajectoryFitter = TrackingTools.TrackFitters.RungeKuttaKFTrajectoryFitterESProducer_cfi.RKTrajectoryFitter.clone()
largeD0step1RKTrajectorySmoother = TrackingTools.TrackFitters.RungeKuttaKFTrajectorySmootherESProducer_cfi.RKTrajectorySmoother.clone()
largeD0step1RKTrajectoryFitter.ComponentName = cms.string('largeD0step1RKFitter')
largeD0step1RKTrajectorySmoother.ComponentName = cms.string('largeD0step1RKSmoother')
largeD0step1RKTrajectoryFitter.minHits = 6
largeD0step1RKTrajectorySmoother.minHits = 6

#TRACKS
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
largeD0step1WithMaterialTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
largeD0step1WithMaterialTracks.src = 'largeD0step1TrackCandidates'
largeD0step1WithMaterialTracks.clusterRemovalInfo = 'largeD0step1Clusters'
largeD0step1WithMaterialTracks.AlgorithmName = cms.string('iter1LargeD0')
largeD0step1WithMaterialTracks.Fitter = 'largeD0step1FittingSmootherWithOutlierRejection'

largeD0step1 = cms.Sequence(trkfilter1*
                          largeD0step1Clusters*
                          largeD0step1PixelRecHits*largeD0step1StripRecHits*
                          largeD0step1Seeds*
                          largeD0step1TrackCandidates*
                          largeD0step1WithMaterialTracks)
                          







