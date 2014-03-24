import FWCore.ParameterSet.Config as cms

#
# Very large impact parameter tracking: Iteration 1 using pixel-triplet seeding
#

#HIT REMOVAL
trkfilter1 = cms.EDProducer("QualityFilter",
    TrackQuality = cms.string('highPurity'),
# Reject hits found in standard iterations                          
    recTracks = cms.InputTag("tobtecStep")
)

largeD0step1Clusters = cms.EDProducer("TrackClusterRemover",
    trajectories = cms.InputTag("trkfilter1"),

# To run this step eliminating hits from standard iterations.
    oldClusterRemovalInfo = cms.InputTag("fifthClusters"),
    pixelClusters = cms.InputTag("fifthClusters"),
    stripClusters = cms.InputTag("fifthClusters"),

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
from TrackingTools.MaterialEffects.Propagators_PtMin09_cff import *

#TRACKER HITS
import RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi
import RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi
largeD0step1PixelRecHits = RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi.siPixelRecHits.clone(
    src = 'largeD0step1Clusters',
    )
largeD0step1StripRecHits = RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi.siStripMatchedRecHits.clone(
    ClusterProducer = 'largeD0step1Clusters',
)
#SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi
largeD0step1LayerTriplets = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.PixelLayerTriplets.clone()
largeD0step1LayerTriplets.BPix.HitProducer = 'largeD0step1PixelRecHits'
largeD0step1LayerTriplets.FPix.HitProducer = 'largeD0step1PixelRecHits'

#SEEDS
import RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff
from RecoPixelVertexing.PixelTriplets.PixelTripletLargeTipGenerator_cfi import *
largeD0step1Seeds = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone()
largeD0step1Seeds.OrderedHitsFactoryPSet.SeedingLayers = 'largeD0step1LayerTriplets'
# Use modified pixel-triplet code that works best for large impact parameters
largeD0step1Seeds.OrderedHitsFactoryPSet.GeneratorPSet = cms.PSet(PixelTripletLargeTipGenerator)
largeD0step1Seeds.SeedCreatorPSet.ComponentName = 'SeedFromConsecutiveHitsTripletOnlyCreator'
largeD0step1Seeds.RegionFactoryPSet.RegionPSet.ptMin = 0.6
largeD0step1Seeds.RegionFactoryPSet.RegionPSet.originRadius = 3.5
largeD0step1Seeds.RegionFactoryPSet.RegionPSet.originHalfLength = 15
#largeD0step1Seeds.SeedCreatorPSet.propagator = cms.string('PropagatorWithMaterialPtMin09')

#TRAJECTORY MEASUREMENT
import RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi
largeD0step1MeasurementTracker = RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi.MeasurementTracker.clone(
    ComponentName = 'largeD0step1MeasurementTracker',
    pixelClusterProducer = 'largeD0step1Clusters',
    stripClusterProducer = 'largeD0step1Clusters',
)
#TRAJECTORY FILTERS (for inwards and outwards track building steps)
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi

largeD0step1CkfTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone(
    ComponentName = 'largeD0step1CkfTrajectoryFilter',
)
#largeD0step1CkfTrajectoryFilter.filterPset.maxLostHits = 1
#largeD0step1CkfTrajectoryFilter.filterPset.maxConsecLostHits = 2
largeD0step1CkfTrajectoryFilter.filterPset.minimumNumberOfHits = 6
largeD0step1CkfTrajectoryFilter.filterPset.minPt = 0.6
largeD0step1CkfTrajectoryFilter.filterPset.minHitsMinPt = 3

#TRAJECTORY BUILDER
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
largeD0step1CkfTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone(
    ComponentName = 'largeD0step1CkfTrajectoryBuilder',
    MeasurementTrackerName = 'largeD0step1MeasurementTracker',
    trajectoryFilterName = 'largeD0step1CkfTrajectoryFilter',
    useSameTrajFilter = True,
    minNrOfHitsForRebuild = 6,
    #lar    maxCand = 5,
    #lar    lostHitPenalty = 100.,
    #lar    alwaysUseInvalidHits = False,
    #lar    propagatorAlong = cms.string('PropagatorWithMaterialPtMin09'),
    #lar    propagatorOpposite = cms.string('PropagatorWithMaterialOppositePtMin09'),
)
#TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
largeD0step1TrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = 'largeD0step1Seeds',
    TrajectoryBuilder = 'largeD0step1CkfTrajectoryBuilder',
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True,
    cleanTrajectoryAfterInOut = True,
)
#
# TRACK FITTING AND SMOOTHING
#

import TrackingTools.TrackFitters.RungeKuttaFitters_cff
largeD0step1FittingSmootherWithOutlierRejection = TrackingTools.TrackFitters.RungeKuttaFitters_cff.RKFittingSmoother.clone(
    ComponentName = 'largeD0step1FittingSmootherWithOutlierRejection',
    EstimateCut = 20,
    MinNumberOfHits = 6,
    Fitter = cms.string('largeD0step1RKFitter'),
    Smoother = cms.string('largeD0step1RKSmoother'),
)
# Also necessary to specify minimum number of hits after final track fit

largeD0step1RKTrajectoryFitter = TrackingTools.TrackFitters.RungeKuttaFitters_cff.RKTrajectoryFitter.clone(
    ComponentName = cms.string('largeD0step1RKFitter'),
    minHits = 6,
)
largeD0step1RKTrajectorySmoother = TrackingTools.TrackFitters.RungeKuttaFitters_cff.RKTrajectorySmoother.clone(
    ComponentName = cms.string('largeD0step1RKSmoother'),
    minHits = 6,
)
#TRACKS
import RecoTracker.TrackProducer.TrackProducer_cfi
largeD0step1WithMaterialTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'largeD0step1TrackCandidates',
    clusterRemovalInfo = 'largeD0step1Clusters',
    AlgorithmName = cms.string('iter1LargeD0'),
    Fitter = 'largeD0step1FittingSmootherWithOutlierRejection',
)
# TRACK QUALITY DEFINITION
import RecoTracker.FinalTrackSelectors.selectLoose_cfi
import RecoTracker.FinalTrackSelectors.selectTight_cfi
import RecoTracker.FinalTrackSelectors.selectHighPurity_cfi

largeD0step1Loose = RecoTracker.FinalTrackSelectors.selectLoose_cfi.selectLoose.clone(
    src = 'largeD0step1WithMaterialTracks',
    keepAllTracks = False,
    copyExtras = False,
    copyTrajectories = True,
    applyAdaptedPVCuts = False,
    chi2n_par = 99.,
    minNumberLayers = 5,
    minNumber3DLayers = 0,
)
largeD0step1Tight = RecoTracker.FinalTrackSelectors.selectTight_cfi.selectTight.clone(
    src = 'largeD0step1Loose',
    keepAllTracks = True,
    copyExtras = False,
    copyTrajectories = True,
    applyAdaptedPVCuts = False,
    chi2n_par = 99.,
    minNumberLayers = 10,
    minNumber3DLayers = 3,
)
largeD0step1Trk = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone(
    src = 'largeD0step1Tight',
    keepAllTracks = True,
    copyExtras = False,
    copyTrajectories = True,
    applyAdaptedPVCuts = False,
    chi2n_par = 99.,
    minNumberLayers = 10,
    minNumber3DLayers = 3,
)
largeD0step1 = cms.Sequence(trkfilter1*
                            largeD0step1Clusters*
                            largeD0step1PixelRecHits*largeD0step1StripRecHits*
                            largeD0step1LayerTriplets*
                            largeD0step1Seeds*
                            largeD0step1TrackCandidates*
                            largeD0step1WithMaterialTracks*
                            largeD0step1Loose*
                            largeD0step1Tight*
                            largeD0step1Trk)
                          







