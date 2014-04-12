import FWCore.ParameterSet.Config as cms

#
# Very large impact parameter tracking: Iteration 5 using TOB + TEC ring 5 seeding
#

#HIT REMOVAL
trkfilter5 = cms.EDProducer("QualityFilter",
    TrackQuality = cms.string('highPurity'),
# Reject hits found in standard iterations                          
    recTracks = cms.InputTag("tobtecStep")
# Reject hits found in all previous iterations                          
#    recTracks = cms.InputTag("largeD0step4")
)

largeD0step5Clusters = cms.EDProducer("TrackClusterRemover",
    trajectories = cms.InputTag("trkfilter5"),

# To run this step eliminating hits from standard iterations.
    oldClusterRemovalInfo = cms.InputTag("fifthClusters"),
    pixelClusters = cms.InputTag("fifthClusters"),
    stripClusters = cms.InputTag("fifthClusters"),

# To run this step, eliminating hits from all previous iterations ...   
#    oldClusterRemovalInfo = cms.InputTag("largeD0step4Clusters"),
#    pixelClusters = cms.InputTag("largeD0step4Clusters"),
#    stripClusters = cms.InputTag("largeD0step4Clusters"),

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

#TRACKER HITS
import RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi
import RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi
largeD0step5PixelRecHits = RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi.siPixelRecHits.clone(
    src = 'largeD0step5Clusters',
)
largeD0step5StripRecHits = RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi.siStripMatchedRecHits.clone(
    ClusterProducer = 'largeD0step5Clusters',
)
#SEEDING LAYERS
import RecoTracker.TkSeedingLayers.TobTecLayerPairs_cfi
largeD0step5LayerPairs = RecoTracker.TkSeedingLayers.TobTecLayerPairs_cfi.TobTecLayerPairs.clone()
largeD0step5LayerPairs.TOB.matchedRecHits = 'largeD0step5StripRecHits:matchedRecHit'
largeD0step5LayerPairs.TEC.matchedRecHits = 'largeD0step5StripRecHits:matchedRecHit'

#SEEDS
from RecoTracker.TkSeedGenerator.GlobalPixelLessSeeds_cff import *
largeD0step5Seeds = RecoTracker.TkSeedGenerator.GlobalPixelLessSeeds_cff.globalPixelLessSeeds.clone()
largeD0step5Seeds.OrderedHitsFactoryPSet.SeedingLayers = 'largeD0step5LayerPairs'
largeD0step5Seeds.RegionFactoryPSet.RegionPSet.ptMin = 0.6
largeD0step5Seeds.RegionFactoryPSet.RegionPSet.originRadius = 10.0
largeD0step5Seeds.RegionFactoryPSet.RegionPSet.originHalfLength = 20.0
#largeD0step5Seeds.SeedCreatorPSet.propagator = cms.string('PropagatorWithMaterialPtMin09')

#TRAJECTORY MEASUREMENT
import RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi
largeD0step5MeasurementTracker = RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi.MeasurementTracker.clone(
    ComponentName = 'largeD0step5MeasurementTracker',
    pixelClusterProducer = 'largeD0step5Clusters',
    stripClusterProducer = 'largeD0step5Clusters',
)
#TRAJECTORY FILTERS (for inwards and outwards track building steps)
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi

largeD0step5CkfTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone(
    ComponentName = 'largeD0step5CkfTrajectoryFilter',
)
largeD0step5CkfTrajectoryFilter.filterPset.maxLostHits = 0
#largeD0step5CkfTrajectoryFilter.filterPset.maxConsecLostHits = 2
largeD0step5CkfTrajectoryFilter.filterPset.minimumNumberOfHits = 6
largeD0step5CkfTrajectoryFilter.filterPset.minPt = 0.6
largeD0step5CkfTrajectoryFilter.filterPset.minHitsMinPt = 3

largeD0step5CkfInOutTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone(
    ComponentName = 'largeD0step5CkfInOutTrajectoryFilter'
    )
largeD0step5CkfInOutTrajectoryFilter.filterPset.maxLostHits = 0
#lar    largeD0step5CkfInOutTrajectoryFilter.filterPset.maxConsecLostHits = 2
largeD0step5CkfInOutTrajectoryFilter.filterPset.minimumNumberOfHits = 4
largeD0step5CkfInOutTrajectoryFilter.filterPset.minPt = 0.6
largeD0step5CkfInOutTrajectoryFilter.filterPset.minHitsMinPt = 3

#TRAJECTORY BUILDER
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
largeD0step5CkfTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone(
    ComponentName = 'largeD0step5CkfTrajectoryBuilder',
    MeasurementTrackerName = 'largeD0step5MeasurementTracker',
    trajectoryFilterName = 'largeD0step5CkfTrajectoryFilter',
    inOutTrajectoryFilterName = 'largeD0step5CkfInOutTrajectoryFilter',
    useSameTrajFilter = False,
    minNrOfHitsForRebuild = 4,
    #lar    maxCand = 5,
    #lar    lostHitPenalty = 100.,
    #lar    alwaysUseInvalidHits = False,
    #lar    propagatorAlong = cms.string('PropagatorWithMaterialPtMin09'),
    #lar    propagatorOpposite = cms.string('PropagatorWithMaterialOppositePtMin09'),
)
#TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
largeD0step5TrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = 'largeD0step5Seeds',
    TrajectoryBuilder = 'largeD0step5CkfTrajectoryBuilder',
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True,
    cleanTrajectoryAfterInOut = False,
)
#
# TRACK FITTING AND SMOOTHING
#

import TrackingTools.TrackFitters.RungeKuttaFitters_cff
largeD0step5FittingSmootherWithOutlierRejection = TrackingTools.TrackFitters.RungeKuttaFitters_cff.RKFittingSmoother.clone(
    ComponentName = 'largeD0step5FittingSmootherWithOutlierRejection',
    EstimateCut = 20,
    MinNumberOfHits = 6,
    Fitter = cms.string('largeD0step5RKFitter'),
    Smoother = cms.string('largeD0step5RKSmoother'),
)
# Also necessary to specify minimum number of hits after final track fit
largeD0step5RKTrajectoryFitter = TrackingTools.TrackFitters.RungeKuttaFitters_cff.RKTrajectoryFitter.clone(
    ComponentName = cms.string('largeD0step5RKFitter'),
    minHits = 6,
)
largeD0step5RKTrajectorySmoother = TrackingTools.TrackFitters.RungeKuttaFitters_cff.RKTrajectorySmoother.clone(
    ComponentName = cms.string('largeD0step5RKSmoother'),
    minHits = 6,
)
#TRACKS
import RecoTracker.TrackProducer.TrackProducer_cfi
largeD0step5WithMaterialTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'largeD0step5TrackCandidates',
    clusterRemovalInfo = 'largeD0step5Clusters',
    AlgorithmName = cms.string('iter5LargeD0'),
    Fitter = 'largeD0step5FittingSmootherWithOutlierRejection',
)
# TRACK QUALITY DEFINITION
import RecoTracker.FinalTrackSelectors.selectLoose_cfi
import RecoTracker.FinalTrackSelectors.selectTight_cfi
import RecoTracker.FinalTrackSelectors.selectHighPurity_cfi

largeD0step5Loose = RecoTracker.FinalTrackSelectors.selectLoose_cfi.selectLoose.clone(
    src = 'largeD0step5WithMaterialTracks',
    keepAllTracks = False,
    copyExtras = False,
    copyTrajectories = True,
    applyAdaptedPVCuts = False,
    chi2n_par = 99.,
    minNumberLayers = 5,
    minNumber3DLayers = 0,
    maxNumberLostLayers = 0,
)
largeD0step5Tight = RecoTracker.FinalTrackSelectors.selectTight_cfi.selectTight.clone(
    src = 'largeD0step5Loose',
    keepAllTracks = True,
    copyExtras = False,
    copyTrajectories = True,
    applyAdaptedPVCuts = False,
    chi2n_par = 99.,
    minNumberLayers = 7,
    minNumber3DLayers = 2,
    maxNumberLostLayers = 0,
)
largeD0step5Trk = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone(
    src = 'largeD0step5Tight',
    keepAllTracks = True,
    copyExtras = False,
    copyTrajectories = True,
    applyAdaptedPVCuts = False,
    chi2n_par = 99.,
    minNumberLayers = 7,
    minNumber3DLayers = 2,
    maxNumberLostLayers = 1,
)
largeD0step5 = cms.Sequence(trkfilter5*
                          largeD0step5Clusters*
                          largeD0step5PixelRecHits*largeD0step5StripRecHits*
                          largeD0step5LayerPairs*
                          largeD0step5Seeds*
                          largeD0step5TrackCandidates*
                          largeD0step5WithMaterialTracks*
                          largeD0step5Loose*
                          largeD0step5Tight*
                          largeD0step5Trk)

                          







