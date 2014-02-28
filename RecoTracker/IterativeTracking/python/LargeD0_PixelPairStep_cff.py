import FWCore.ParameterSet.Config as cms

#
# Very large impact parameter tracking: Iteration 2 using pixel-pair seeding
#
 
#HIT REMOVAL

trkfilter2 = cms.EDProducer("QualityFilter",
    TrackQuality = cms.string('highPurity'),
# Reject hits found in standard iterations                          
    recTracks = cms.InputTag("tobtecStep")
# Reject hits found in all previous iterations                          
#    recTracks = cms.InputTag("largeD0step1")
)

largeD0step2Clusters = cms.EDProducer("TrackClusterRemover",
    trajectories = cms.InputTag("trkfilter2"),

# To run this step eliminating hits from standard iterations.
    oldClusterRemovalInfo = cms.InputTag("fifthClusters"),
    pixelClusters = cms.InputTag("fifthClusters"),
    stripClusters = cms.InputTag("fifthClusters"),

# To run this step, eliminating hits from all previous iterations ...   
#    oldClusterRemovalInfo = cms.InputTag("largeD0step1Clusters"),
#    pixelClusters = cms.InputTag("largeD0step1Clusters"),
#    stripClusters = cms.InputTag("largeD0step1Clusters"),

# To run it, not eliminating any hits.
#    trajectories = cms.InputTag("zeroStepFilter"),
#    pixelClusters = cms.InputTag("siPixelClusters"),
#    stripClusters = cms.InputTag("siStripClusters"),
                                     
    Common = cms.PSet(
       maxChi2 = cms.double(30.0)
# To run it not eliminating any hits, you also need ...
#       maxChi2 = cms.double(0.0)
    )
)

# Propagator taking into account momentum uncertainty in multiple
# scattering calculation.
#from TrackingTools.MaterialEffects.Propagators_PtMin09_cff import *
import TrackingTools.MaterialEffects.MaterialPropagator_cfi
MaterialPropagatorPtMin06 = TrackingTools.MaterialEffects.MaterialPropagator_cfi.MaterialPropagator.clone(
    ComponentName = 'PropagatorWithMaterialPtMin06',
    ptMin = 0.6
    )
 
import TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi
OppositeMaterialPropagatorPtMin06 = TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi.OppositeMaterialPropagator.clone(
    ComponentName = 'PropagatorWithMaterialOppositePtMin06',
    ptMin = 0.6
    )

#TRACKER HITS
import RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi
import RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi
largeD0step2PixelRecHits = RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi.siPixelRecHits.clone(
    src = 'largeD0step2Clusters',
    )
largeD0step2StripRecHits = RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi.siStripMatchedRecHits.clone(
    ClusterProducer = 'largeD0step2Clusters',
    )

#SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelLayerPairs_cfi
largeD0step2LayerPairs = RecoTracker.TkSeedingLayers.PixelLayerPairs_cfi.PixelLayerPairs.clone()
largeD0step2LayerPairs.BPix.HitProducer = 'largeD0step2PixelRecHits'
largeD0step2LayerPairs.FPix.HitProducer = 'largeD0step2PixelRecHits'

#SEEDS
import RecoTracker.TkSeedGenerator.GlobalPixelSeeds_cff
largeD0step2Seeds = RecoTracker.TkSeedGenerator.GlobalPixelSeeds_cff.globalPixelSeeds.clone()
largeD0step2Seeds.OrderedHitsFactoryPSet.SeedingLayers = 'largeD0step2LayerPairs'
largeD0step2Seeds.RegionFactoryPSet.RegionPSet.ptMin = 0.6
largeD0step2Seeds.RegionFactoryPSet.RegionPSet.originRadius = 2.5
largeD0step2Seeds.RegionFactoryPSet.RegionPSet.originHalfLength = 15
import RecoTracker.TkSeedGenerator.SeedFromConsecutiveHitsStraightLineCreator_cfi
largeD0step2Seeds.SeedCreatorPSet = RecoTracker.TkSeedGenerator.SeedFromConsecutiveHitsStraightLineCreator_cfi.SeedFromConsecutiveHitsStraightLineCreator.clone(
    propagator = cms.string('PropagatorWithMaterialPtMin06')
)


#TRAJECTORY MEASUREMENT
import RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi
largeD0step2MeasurementTracker = RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi.MeasurementTracker.clone(
    ComponentName = 'largeD0step2MeasurementTracker',
    pixelClusterProducer = 'largeD0step2Clusters',
    stripClusterProducer = 'largeD0step2Clusters'
    )

#TRAJECTORY FILTERS (for inwards and outwards track building steps)
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi

largeD0step2CkfTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone(
    ComponentName = 'largeD0step2CkfTrajectoryFilter'
    )
#largeD0step2CkfTrajectoryFilter.filterPset.maxLostHits = 1
#largeD0step2CkfTrajectoryFilter.filterPset.maxConstep2LostHits = 2
largeD0step2CkfTrajectoryFilter.filterPset.minimumNumberOfHits = 6
largeD0step2CkfTrajectoryFilter.filterPset.minPt = 0.6
largeD0step2CkfTrajectoryFilter.filterPset.minHitsMinPt = 3

#TRAJECTORY BUILDER
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
largeD0step2CkfTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone(
    ComponentName = 'largeD0step2CkfTrajectoryBuilder',
    MeasurementTrackerName = 'largeD0step2MeasurementTracker',
    trajectoryFilterName = 'largeD0step2CkfTrajectoryFilter',
    useSameTrajFilter = True,
    minNrOfHitsForRebuild = 6,
    maxCand = 5,
    #lostHitPenalty = 100.,
    #alwaysUseInvalidHits = False,
    propagatorAlong = cms.string('PropagatorWithMaterialPtMin06'),
    propagatorOpposite = cms.string('PropagatorWithMaterialOppositePtMin06')
    )

#TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
largeD0step2TrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = 'largeD0step2Seeds',
    TrajectoryBuilder = 'largeD0step2CkfTrajectoryBuilder',
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True,
    cleanTrajectoryAfterInOut = True
    )

#
# TRACK FITTING AND SMOOTHING
#

import TrackingTools.TrackFitters.RungeKuttaFitters_cff
largeD0step2FittingSmootherWithOutlierRejection = TrackingTools.TrackFitters.RungeKuttaFitters_cff.RKFittingSmoother.clone(
    ComponentName = 'largeD0step2FittingSmootherWithOutlierRejection',
    EstimateCut = 20,
    MinNumberOfHits = 6,
    Fitter = cms.string('largeD0step2RKFitter'),
    Smoother = cms.string('largeD0step2RKSmoother'),
)
# Also necessary to specify minimum number of hits after final track fit
largeD0step2RKTrajectoryFitter = TrackingTools.TrackFitters.RungeKuttaFitters_cff.RKTrajectoryFitter.clone(
    ComponentName = cms.string('largeD0step2RKFitter'),
    minHits = 6,
)
largeD0step2RKTrajectorySmoother = TrackingTools.TrackFitters.RungeKuttaFitters_cff.RKTrajectorySmoother.clone(
    ComponentName = cms.string('largeD0step2RKSmoother'),
    minHits = 6,
)
#TRACKS
import RecoTracker.TrackProducer.TrackProducer_cfi
largeD0step2WithMaterialTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'largeD0step2TrackCandidates',
    clusterRemovalInfo = 'largeD0step2Clusters',
    AlgorithmName = cms.string('iter2LargeD0'),
    Fitter = 'largeD0step2FittingSmootherWithOutlierRejection',
    )

# TRACK QUALITY DEFINITION
import RecoTracker.FinalTrackSelectors.selectLoose_cfi
import RecoTracker.FinalTrackSelectors.selectTight_cfi
import RecoTracker.FinalTrackSelectors.selectHighPurity_cfi

largeD0step2Loose = RecoTracker.FinalTrackSelectors.selectLoose_cfi.selectLoose.clone(
    src = 'largeD0step2WithMaterialTracks',
    keepAllTracks = False,
    copyExtras = False,
    copyTrajectories = True,
    applyAdaptedPVCuts = False,
    chi2n_par = 99.,
    minNumberLayers = 5,
    minNumber3DLayers = 0,
    )
largeD0step2Tight = RecoTracker.FinalTrackSelectors.selectTight_cfi.selectTight.clone(
    src = 'largeD0step2Loose',
    keepAllTracks = True,
    copyExtras = False,
    copyTrajectories = True,
    applyAdaptedPVCuts = False,
    chi2n_par = 99.,
    minNumberLayers = 10,
    minNumber3DLayers = 3,
    )

largeD0step2Trk = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone(
    src = 'largeD0step2Tight',
    keepAllTracks = True,
    copyExtras = False,
    copyTrajectories = True,
    applyAdaptedPVCuts = False,
    chi2n_par = 99.,
    minNumberLayers = 10,
    minNumber3DLayers = 3,
    )

largeD0step2 = cms.Sequence(trkfilter2*
                            largeD0step2Clusters*
                            largeD0step2PixelRecHits*largeD0step2StripRecHits*
                            largeD0step2LayerPairs*
                            largeD0step2Seeds*
                            largeD0step2TrackCandidates*
                            largeD0step2WithMaterialTracks*
                            largeD0step2Loose*
                            largeD0step2Tight*
                            largeD0step2Trk)
                          






