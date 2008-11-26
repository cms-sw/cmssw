import FWCore.ParameterSet.Config as cms

#
# Very large impact parameter tracking: Iteration 3 using Outer pixel + inner TIB/TID/TEC ring seeding
#

#HIT REMOVAL
trkfilter3 = cms.EDFilter("QualityFilter",
    TrackQuality = cms.string('highPurity'),
    recTracks = cms.InputTag("tobtecStep")
)

largeD0step3Clusters = cms.EDFilter("TrackClusterRemover",
# To run this step, eliminating hits from all previous iterations ...                                    
#    trajectories = cms.InputTag("largeD0step2"),
#    oldClusterRemovalInfo = cms.InputTag("largeD0step2Clusters"),
#    pixelClusters = cms.InputTag("largeD0step2Clusters"),
#    stripClusters = cms.InputTag("largeD0step2Clusters"),

# To run this step independently of the other large d0 tracking iterations ...
#    trajectories = cms.InputTag("tobtecStep"),
    trajectories = cms.InputTag("trkfilter3"),
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
#       maxChi2 = cms.double(0.0)
    )
)

# Propagator taking into account momentum uncertainty in multiple
# scattering calculation.
#from TrackingTools.MaterialEffects.Propagators_PtMin09_cff import *
import TrackingTools.MaterialEffects.MaterialPropagator_cfi
MaterialPropagatorPtMin06 = TrackingTools.MaterialEffects.MaterialPropagator_cfi.MaterialPropagator.clone()
MaterialPropagatorPtMin06.ComponentName = 'PropagatorWithMaterialPtMin06'
MaterialPropagatorPtMin06.ptMin = 0.6
 
import TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi
OppositeMaterialPropagatorPtMin06 = TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi.OppositeMaterialPropagator.clone()
OppositeMaterialPropagatorPtMin06.ComponentName = 'PropagatorWithMaterialOppositePtMin06'
OppositeMaterialPropagatorPtMin06.ptMin = 0.6

#TRACKER HITS
import RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi
import RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi
largeD0step3PixelRecHits = RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi.siPixelRecHits.clone()
largeD0step3StripRecHits = RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi.siStripMatchedRecHits.clone()
largeD0step3PixelRecHits.src = 'largeD0step3Clusters'
largeD0step3StripRecHits.ClusterProducer = 'largeD0step3Clusters'

#SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelAndStripLayerPairs_cfi
largeD0step3layerpairs = RecoTracker.TkSeedingLayers.PixelAndStripLayerPairs_cfi.pixelandstriplayerpairs.clone()
largeD0step3layerpairs.ComponentName = 'largeD0step3LayerPairs'
largeD0step3layerpairs.BPix.HitProducer = 'largeD0step3PixelRecHits'
largeD0step3layerpairs.FPix.HitProducer = 'largeD0step3PixelRecHits'
largeD0step3layerpairs.TIB.matchedRecHits = 'largeD0step3StripRecHits:matchedRecHit'
largeD0step3layerpairs.TID.matchedRecHits = 'largeD0step3StripRecHits:matchedRecHit'
largeD0step3layerpairs.TEC.matchedRecHits = 'largeD0step3StripRecHits:matchedRecHit'

#SEEDS
from RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cff import *
import RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cfi
largeD0step3Seeds = RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cfi.globalMixedSeeds.clone()
largeD0step3Seeds.OrderedHitsFactoryPSet.SeedingLayers = 'largeD0step3LayerPairs'
largeD0step3Seeds.RegionFactoryPSet.RegionPSet.ptMin = 0.6
largeD0step3Seeds.RegionFactoryPSet.RegionPSet.originRadius = 3.5
largeD0step3Seeds.RegionFactoryPSet.RegionPSet.originHalfLength = 12.5
largeD0step3Seeds.propagator = cms.string('PropagatorWithMaterialPtMin06')
# The fast-helix fit doesn't work well for large d0 pixel pair seeding.
largeD0step3Seeds.UseFastHelix = False


#TRAJECTORY MEASUREMENT
import RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi
largeD0step3MeasurementTracker = RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi.MeasurementTracker.clone()
largeD0step3MeasurementTracker.ComponentName = 'largeD0step3MeasurementTracker'
largeD0step3MeasurementTracker.pixelClusterProducer = 'largeD0step3Clusters'
largeD0step3MeasurementTracker.stripClusterProducer = 'largeD0step3Clusters'

#TRAJECTORY FILTERS (for inwards and outwards track building steps)
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi

largeD0step3CkfTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone()
largeD0step3CkfTrajectoryFilter.ComponentName = 'largeD0step3CkfTrajectoryFilter'
largeD0step3CkfTrajectoryFilter.filterPset.maxLostHits = 0
#largeD0step3CkfTrajectoryFilter.filterPset.maxConsecLostHits = 2
largeD0step3CkfTrajectoryFilter.filterPset.minimumNumberOfHits = 7
largeD0step3CkfTrajectoryFilter.filterPset.minPt = 0.6
largeD0step3CkfTrajectoryFilter.filterPset.minHitsMinPt = 3

largeD0step3CkfInOutTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone()
largeD0step3CkfInOutTrajectoryFilter.ComponentName = 'largeD0step3CkfInOutTrajectoryFilter'
largeD0step3CkfInOutTrajectoryFilter.filterPset.maxLostHits = 0
#largeD0step3CkfInOutTrajectoryFilter.filterPset.maxConsecLostHits = 2
largeD0step3CkfInOutTrajectoryFilter.filterPset.minimumNumberOfHits = 7
largeD0step3CkfInOutTrajectoryFilter.filterPset.minPt = 0.6
largeD0step3CkfInOutTrajectoryFilter.filterPset.minHitsMinPt = 3

#TRAJECTORY BUILDER
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
largeD0step3CkfTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone()
largeD0step3CkfTrajectoryBuilder.ComponentName = 'largeD0step3CkfTrajectoryBuilder'
largeD0step3CkfTrajectoryBuilder.MeasurementTrackerName = 'largeD0step3MeasurementTracker'
largeD0step3CkfTrajectoryBuilder.trajectoryFilterName = 'largeD0step3CkfTrajectoryFilter'
largeD0step3CkfTrajectoryBuilder.inOutTrajectoryFilterName = 'largeD0step3CkfInOutTrajectoryFilter'
largeD0step3CkfTrajectoryBuilder.useSameTrajFilter = False
largeD0step3CkfTrajectoryBuilder.minNrOfHitsForRebuild = 7
#largeD0step3CkfTrajectoryBuilder.maxCand = 5
#largeD0step3CkfTrajectoryBuilder.lostHitPenalty = 100.
#largeD0step3CkfTrajectoryBuilder.alwaysUseInvalidHits = False
largeD0step3CkfTrajectoryBuilder.propagatorAlong = cms.string('PropagatorWithMaterialPtMin06')
largeD0step3CkfTrajectoryBuilder.propagatorOpposite = cms.string('PropagatorWithMaterialOppositePtMin06')

#TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
largeD0step3TrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone()
largeD0step3TrackCandidates.SeedProducer = 'largeD0step3Seeds'
largeD0step3TrackCandidates.TrajectoryBuilder = 'largeD0step3CkfTrajectoryBuilder'
largeD0step3TrackCandidates.doSeedingRegionRebuilding = True
largeD0step3TrackCandidates.useHitsSplitting = True
largeD0step3TrackCandidates.cleanTrajectoryAfterInOut = True

#
# TRACK FITTING AND SMOOTHING
#

import TrackingTools.TrackFitters.RungeKuttaKFFittingSmootherESProducer_cfi
largeD0step3FittingSmootherWithOutlierRejection = TrackingTools.TrackFitters.RungeKuttaKFFittingSmootherESProducer_cfi.RKFittingSmoother.clone()
largeD0step3FittingSmootherWithOutlierRejection.ComponentName = 'largeD0step3FittingSmootherWithOutlierRejection'
largeD0step3FittingSmootherWithOutlierRejection.EstimateCut = 20
largeD0step3FittingSmootherWithOutlierRejection.MinNumberOfHits = 7
largeD0step3FittingSmootherWithOutlierRejection.Fitter = cms.string('largeD0step3RKFitter')
largeD0step3FittingSmootherWithOutlierRejection.Smoother = cms.string('largeD0step3RKSmoother')

# Also necessary to specify minimum number of hits after final track fit
import TrackingTools.TrackFitters.RungeKuttaKFTrajectoryFitterESProducer_cfi
import TrackingTools.TrackFitters.RungeKuttaKFTrajectorySmootherESProducer_cfi
largeD0step3RKTrajectoryFitter = TrackingTools.TrackFitters.RungeKuttaKFTrajectoryFitterESProducer_cfi.RKTrajectoryFitter.clone()
largeD0step3RKTrajectorySmoother = TrackingTools.TrackFitters.RungeKuttaKFTrajectorySmootherESProducer_cfi.RKTrajectorySmoother.clone()
largeD0step3RKTrajectoryFitter.ComponentName = cms.string('largeD0step3RKFitter')
largeD0step3RKTrajectorySmoother.ComponentName = cms.string('largeD0step3RKSmoother')
largeD0step3RKTrajectoryFitter.minHits = 7
largeD0step3RKTrajectorySmoother.minHits = 7

#TRACKS
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
largeD0step3WithMaterialTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
largeD0step3WithMaterialTracks.src = 'largeD0step3TrackCandidates'
largeD0step3WithMaterialTracks.clusterRemovalInfo = 'largeD0step3Clusters'
largeD0step3WithMaterialTracks.AlgorithmName = cms.string('iter3LargeD0')
largeD0step3WithMaterialTracks.Fitter = 'largeD0step3FittingSmootherWithOutlierRejection'

largeD0step3 = cms.Sequence(trkfilter3*
                          largeD0step3Clusters*
                          largeD0step3PixelRecHits*largeD0step3StripRecHits*
                          largeD0step3Seeds*
                          largeD0step3TrackCandidates*
                          largeD0step3WithMaterialTracks)
                          







