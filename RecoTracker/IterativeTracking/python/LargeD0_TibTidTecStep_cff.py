import FWCore.ParameterSet.Config as cms

#
# Very large impact parameter tracking: Iteration 4 using TIB + TEC ring 1-2 seeding
#

#HIT REMOVAL
trkfilter4 = cms.EDProducer("QualityFilter",
    TrackQuality = cms.string('highPurity'),
# Reject hits found in standard iterations                          
    recTracks = cms.InputTag("tobtecStep")
# Reject hits found in all previous iterations                          
#    recTracks = cms.InputTag("largeD0step3")
)

largeD0step4Clusters = cms.EDProducer("TrackClusterRemover",
    trajectories = cms.InputTag("trkfilter4"),

# To run this step eliminating hits from standard iterations.
    oldClusterRemovalInfo = cms.InputTag("fifthClusters"),
    pixelClusters = cms.InputTag("fifthClusters"),
    stripClusters = cms.InputTag("fifthClusters"),

# To run this step, eliminating hits from all previous iterations ...   
#    oldClusterRemovalInfo = cms.InputTag("largeD0step3Clusters"),
#    pixelClusters = cms.InputTag("largeD0step3Clusters"),
#    stripClusters = cms.InputTag("largeD0step3Clusters"),

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
largeD0step4PixelRecHits = RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi.siPixelRecHits.clone(
    src = 'largeD0step4Clusters',
)
largeD0step4StripRecHits = RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi.siStripMatchedRecHits.clone(
    ClusterProducer = 'largeD0step4Clusters',
)
#SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelLessLayerPairs_cfi
largeD0step4LayerPairs = RecoTracker.TkSeedingLayers.PixelLessLayerPairs_cfi.PixelLessLayerPairs.clone()
largeD0step4LayerPairs.TIB.matchedRecHits = 'largeD0step4StripRecHits:matchedRecHit'
largeD0step4LayerPairs.TID.matchedRecHits = 'largeD0step4StripRecHits:matchedRecHit'
largeD0step4LayerPairs.TEC.matchedRecHits = 'largeD0step4StripRecHits:matchedRecHit'

#SEEDS
import RecoTracker.TkSeedGenerator.GlobalPixelLessSeeds_cff
largeD0step4Seeds = RecoTracker.TkSeedGenerator.GlobalPixelLessSeeds_cff.globalPixelLessSeeds.clone()
largeD0step4Seeds.OrderedHitsFactoryPSet.SeedingLayers = 'largeD0step4LayerPairs'
largeD0step4Seeds.RegionFactoryPSet.RegionPSet.ptMin = 0.6
largeD0step4Seeds.RegionFactoryPSet.RegionPSet.originRadius = 5.0
largeD0step4Seeds.RegionFactoryPSet.RegionPSet.originHalfLength = 15.0
#largeD0step4Seeds.SeedCreatorPSet.propagator = cms.string('PropagatorWithMaterialPtMin09')

#TRAJECTORY MEASUREMENT
import RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi
largeD0step4MeasurementTracker = RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi.MeasurementTracker.clone(
    ComponentName = 'largeD0step4MeasurementTracker',
    pixelClusterProducer = 'largeD0step4Clusters',
    stripClusterProducer = 'largeD0step4Clusters',
)
#TRAJECTORY FILTERS (for inwards and outwards track building steps)
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi

largeD0step4CkfTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone(
    ComponentName = 'largeD0step4CkfTrajectoryFilter',
)
largeD0step4CkfTrajectoryFilter.filterPset.maxLostHits = 0
#largeD0step4CkfTrajectoryFilter.filterPset.maxConsecLostHits = 2
largeD0step4CkfTrajectoryFilter.filterPset.minimumNumberOfHits = 7
largeD0step4CkfTrajectoryFilter.filterPset.minPt = 0.6
largeD0step4CkfTrajectoryFilter.filterPset.minHitsMinPt = 3

largeD0step4CkfInOutTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone(
    ComponentName = 'largeD0step4CkfInOutTrajectoryFilter',
)
largeD0step4CkfInOutTrajectoryFilter.filterPset.maxLostHits = 0
#largeD0step4CkfInOutTrajectoryFilter.filterPset.maxConsecLostHits = 2
largeD0step4CkfInOutTrajectoryFilter.filterPset.minimumNumberOfHits = 7
largeD0step4CkfInOutTrajectoryFilter.filterPset.minPt = 0.6
largeD0step4CkfInOutTrajectoryFilter.filterPset.minHitsMinPt = 3

#TRAJECTORY BUILDER
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
largeD0step4CkfTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone(
    ComponentName = 'largeD0step4CkfTrajectoryBuilder',
    MeasurementTrackerName = 'largeD0step4MeasurementTracker',
    trajectoryFilterName = 'largeD0step4CkfTrajectoryFilter',
    inOutTrajectoryFilterName = 'largeD0step4CkfInOutTrajectoryFilter',
    useSameTrajFilter = False,
    minNrOfHitsForRebuild = 7,
    #lar    maxCand = 5,
    #lar    lostHitPenalty = 100.,
    #lar    alwaysUseInvalidHits = False,
    #lar    propagatorAlong = cms.string('PropagatorWithMaterialPtMin09'),
    #lar    propagatorOpposite = cms.string('PropagatorWithMaterialOppositePtMin09'),
)
#TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
largeD0step4TrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = 'largeD0step4Seeds',
    TrajectoryBuilder = 'largeD0step4CkfTrajectoryBuilder',
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True,
    cleanTrajectoryAfterInOut = True,
)
#
# TRACK FITTING AND SMOOTHING
#

import TrackingTools.TrackFitters.RungeKuttaFitters_cff
largeD0step4FittingSmootherWithOutlierRejection = TrackingTools.TrackFitters.RungeKuttaFitters_cff.RKFittingSmoother.clone(
    ComponentName = 'largeD0step4FittingSmootherWithOutlierRejection',
    EstimateCut = 20,
    MinNumberOfHits = 7,
    Fitter = cms.string('largeD0step4RKFitter'),
    Smoother = cms.string('largeD0step4RKSmoother'),
)
# Also necessary to specify minimum number of hits after final track fit
largeD0step4RKTrajectoryFitter = TrackingTools.TrackFitters.RungeKuttaFitters_cff.RKTrajectoryFitter.clone(
    ComponentName = cms.string('largeD0step4RKFitter'),
    minHits = 7,
)
largeD0step4RKTrajectorySmoother = TrackingTools.TrackFitters.RungeKuttaFitters_cff.RKTrajectorySmoother.clone(
    ComponentName = cms.string('largeD0step4RKSmoother'),
    minHits = 7,
)
#TRACKS
import RecoTracker.TrackProducer.TrackProducer_cfi
largeD0step4WithMaterialTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'largeD0step4TrackCandidates',
    clusterRemovalInfo = 'largeD0step4Clusters',
    AlgorithmName = cms.string('iter4LargeD0'),
    Fitter = 'largeD0step4FittingSmootherWithOutlierRejection',
)
# TRACK QUALITY DEFINITION
import RecoTracker.FinalTrackSelectors.selectLoose_cfi
import RecoTracker.FinalTrackSelectors.selectTight_cfi
import RecoTracker.FinalTrackSelectors.selectHighPurity_cfi

largeD0step4Loose = RecoTracker.FinalTrackSelectors.selectLoose_cfi.selectLoose.clone(
    src = 'largeD0step4WithMaterialTracks',
    keepAllTracks = False,
    copyExtras = False,
    copyTrajectories = True,
    applyAdaptedPVCuts = False,
    chi2n_par = 99.,
    minNumberLayers = 5,
    minNumber3DLayers = 0,
)
largeD0step4Tight = RecoTracker.FinalTrackSelectors.selectTight_cfi.selectTight.clone(
    src = 'largeD0step4Loose',
    keepAllTracks = True,
    copyExtras = False,
    copyTrajectories = True,
    applyAdaptedPVCuts = False,
    chi2n_par = 99.,
    minNumberLayers = 8,
    minNumber3DLayers = 2,
)
largeD0step4Trk = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone(
    src = 'largeD0step4Tight',
    keepAllTracks = True,
    copyExtras = False,
    copyTrajectories = True,
    applyAdaptedPVCuts = False,
    chi2n_par = 99.,
    minNumberLayers = 8,
    minNumber3DLayers = 3,
)
largeD0step4 = cms.Sequence(trkfilter4*
                            largeD0step4Clusters*
                            largeD0step4PixelRecHits*largeD0step4StripRecHits*
                            largeD0step4LayerPairs*
                            largeD0step4Seeds*
                            largeD0step4TrackCandidates*
                            largeD0step4WithMaterialTracks*
                            largeD0step4Loose*
                            largeD0step4Tight*
                            largeD0step4Trk)
                          







