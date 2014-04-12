import FWCore.ParameterSet.Config as cms

#
# Very large impact parameter tracking: Iteration 3 using Outer pixel + inner TIB/TID/TEC ring seeding
#

#HIT REMOVAL
trkfilter3 = cms.EDProducer("QualityFilter",
    TrackQuality = cms.string('highPurity'),
# Reject hits found in standard iterations                          
    recTracks = cms.InputTag("tobtecStep")
# Reject hits found in all previous iterations                          
#    recTracks = cms.InputTag("largeD0step2")
)

largeD0step3Clusters = cms.EDProducer("TrackClusterRemover",
    trajectories = cms.InputTag("trkfilter3"),

# To run this step eliminating hits from standard iterations.
    oldClusterRemovalInfo = cms.InputTag("fifthClusters"),
    pixelClusters = cms.InputTag("fifthClusters"),
    stripClusters = cms.InputTag("fifthClusters"),

# To run this step, eliminating hits from all previous iterations ...   
#    oldClusterRemovalInfo = cms.InputTag("largeD0step2Clusters"),
#    pixelClusters = cms.InputTag("largeD0step2Clusters"),
#    stripClusters = cms.InputTag("largeD0step2Clusters"),

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
    ptMin = 0.6,
) 
import TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi
OppositeMaterialPropagatorPtMin06 = TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi.OppositeMaterialPropagator.clone(
    ComponentName = 'PropagatorWithMaterialOppositePtMin06',
    ptMin = 0.6,
)
#TRACKER HITS
import RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi
import RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi
largeD0step3PixelRecHits = RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi.siPixelRecHits.clone(
    src = 'largeD0step3Clusters',
    )
largeD0step3StripRecHits = RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi.siStripMatchedRecHits.clone(
    ClusterProducer = 'largeD0step3Clusters',
    )
#SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelAndStripLayerPairs_cfi
largeD0step3LayerPairs = RecoTracker.TkSeedingLayers.PixelAndStripLayerPairs_cfi.PixelAndStripLayerPairs.clone()
largeD0step3LayerPairs.BPix.HitProducer = 'largeD0step3PixelRecHits'
largeD0step3LayerPairs.FPix.HitProducer = 'largeD0step3PixelRecHits'
largeD0step3LayerPairs.TIB.matchedRecHits = 'largeD0step3StripRecHits:matchedRecHit'
largeD0step3LayerPairs.TID.matchedRecHits = 'largeD0step3StripRecHits:matchedRecHit'
largeD0step3LayerPairs.TEC.matchedRecHits = 'largeD0step3StripRecHits:matchedRecHit'

#SEEDS
import RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cff
largeD0step3Seeds = RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cff.globalMixedSeeds.clone()
largeD0step3Seeds.OrderedHitsFactoryPSet.SeedingLayers = 'largeD0step3LayerPairs'
largeD0step3Seeds.RegionFactoryPSet.RegionPSet.ptMin = 0.6
largeD0step3Seeds.RegionFactoryPSet.RegionPSet.originRadius = 3.5
largeD0step3Seeds.RegionFactoryPSet.RegionPSet.originHalfLength = 12.5
import RecoTracker.TkSeedGenerator.SeedFromConsecutiveHitsStraightLineCreator_cfi
largeD0step3Seeds.SeedCreatorPSet = RecoTracker.TkSeedGenerator.SeedFromConsecutiveHitsStraightLineCreator_cfi.SeedFromConsecutiveHitsStraightLineCreator.clone(
    propagator = cms.string('PropagatorWithMaterialPtMin06')
)


#TRAJECTORY MEASUREMENT
import RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi
largeD0step3MeasurementTracker = RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi.MeasurementTracker.clone(
    ComponentName = 'largeD0step3MeasurementTracker',
    pixelClusterProducer = 'largeD0step3Clusters',
    stripClusterProducer = 'largeD0step3Clusters',
)
#TRAJECTORY FILTERS (for inwards and outwards track building steps)
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi

largeD0step3CkfTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone(
    ComponentName = 'largeD0step3CkfTrajectoryFilter'
    )
largeD0step3CkfTrajectoryFilter.filterPset.maxLostHits = 0
#lar    largeD0step3CkfTrajectoryFilter.filterPset.maxConsecLostHits = 2
largeD0step3CkfTrajectoryFilter.filterPset.minimumNumberOfHits = 7
largeD0step3CkfTrajectoryFilter.filterPset.minPt = 0.6
largeD0step3CkfTrajectoryFilter.filterPset.minHitsMinPt = 3

largeD0step3CkfInOutTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone(
    ComponentName = 'largeD0step3CkfInOutTrajectoryFilter'
    )
largeD0step3CkfInOutTrajectoryFilter.filterPset.maxLostHits = 0
#lar    largeD0step3CkfInOutTrajectoryFilter.filterPset.maxConsecLostHits = 2
largeD0step3CkfInOutTrajectoryFilter.filterPset.minimumNumberOfHits = 7
largeD0step3CkfInOutTrajectoryFilter.filterPset.minPt = 0.6
largeD0step3CkfInOutTrajectoryFilter.filterPset.minHitsMinPt = 3

#TRAJECTORY BUILDER
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
largeD0step3CkfTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone(
    ComponentName = 'largeD0step3CkfTrajectoryBuilder',
    MeasurementTrackerName = 'largeD0step3MeasurementTracker',
    trajectoryFilterName = 'largeD0step3CkfTrajectoryFilter',
    inOutTrajectoryFilterName = 'largeD0step3CkfInOutTrajectoryFilter',
    useSameTrajFilter = False,
    minNrOfHitsForRebuild = 7,
    #lar    maxCand = 5,
    #lar    lostHitPenalty = 100.,
    #lar    alwaysUseInvalidHits = False,
    propagatorAlong = cms.string('PropagatorWithMaterialPtMin06'),
    propagatorOpposite = cms.string('PropagatorWithMaterialOppositePtMin06'),
)
#TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
largeD0step3TrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = 'largeD0step3Seeds',
    TrajectoryBuilder = 'largeD0step3CkfTrajectoryBuilder',
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True,
    cleanTrajectoryAfterInOut = True,
)
#
# TRACK FITTING AND SMOOTHING
#

import TrackingTools.TrackFitters.RungeKuttaFitters_cff
largeD0step3FittingSmootherWithOutlierRejection = TrackingTools.TrackFitters.RungeKuttaFitters_cff.RKFittingSmoother.clone(
    ComponentName = 'largeD0step3FittingSmootherWithOutlierRejection',
    EstimateCut = 20,
    MinNumberOfHits = 7,
    Fitter = cms.string('largeD0step3RKFitter'),
    Smoother = cms.string('largeD0step3RKSmoother'),
)
# Also necessary to specify minimum number of hits after final track fit
largeD0step3RKTrajectoryFitter = TrackingTools.TrackFitters.RungeKuttaFitters_cff.RKTrajectoryFitter.clone(
    ComponentName = cms.string('largeD0step3RKFitter'),
    minHits = 7,
)
largeD0step3RKTrajectorySmoother = TrackingTools.TrackFitters.RungeKuttaFitters_cff.RKTrajectorySmoother.clone(
    ComponentName = cms.string('largeD0step3RKSmoother'),
    minHits = 7,
)
#TRACKS
import RecoTracker.TrackProducer.TrackProducer_cfi
largeD0step3WithMaterialTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'largeD0step3TrackCandidates',
    clusterRemovalInfo = 'largeD0step3Clusters',
    AlgorithmName = cms.string('iter3LargeD0'),
    Fitter = 'largeD0step3FittingSmootherWithOutlierRejection',
)
# TRACK QUALITY DEFINITION
import RecoTracker.FinalTrackSelectors.selectLoose_cfi
import RecoTracker.FinalTrackSelectors.selectTight_cfi
import RecoTracker.FinalTrackSelectors.selectHighPurity_cfi

largeD0step3Loose = RecoTracker.FinalTrackSelectors.selectLoose_cfi.selectLoose.clone(
    src = 'largeD0step3WithMaterialTracks',
    keepAllTracks = False,
    copyExtras = False,
    copyTrajectories = True,
    applyAdaptedPVCuts = False,
    chi2n_par = 99.,
    minNumberLayers = 5,
    minNumber3DLayers = 0,
)
largeD0step3Tight = RecoTracker.FinalTrackSelectors.selectTight_cfi.selectTight.clone(
    src = 'largeD0step3Loose',
    keepAllTracks = True,
    copyExtras = False,
    copyTrajectories = True,
    applyAdaptedPVCuts = False,
    chi2n_par = 99.,
    minNumberLayers = 10,
    minNumber3DLayers = 3,
)
largeD0step3Trk = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone(
    src = 'largeD0step3Tight',
    keepAllTracks = True,
    copyExtras = False,
    copyTrajectories = True,
    applyAdaptedPVCuts = False,
    chi2n_par = 99.,
    minNumberLayers = 10,
    minNumber3DLayers = 3,
)
largeD0step3 = cms.Sequence(trkfilter3*
                            largeD0step3Clusters*
                            largeD0step3PixelRecHits*largeD0step3StripRecHits*
                            largeD0step3LayerPairs*
                            largeD0step3Seeds*
                            largeD0step3TrackCandidates*
                            largeD0step3WithMaterialTracks*
                            largeD0step3Loose*
                            largeD0step3Tight*
                            largeD0step3Trk)

                          







