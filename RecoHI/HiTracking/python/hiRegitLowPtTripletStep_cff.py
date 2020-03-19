import FWCore.ParameterSet.Config as cms

# pp iterative tracking modified for hiOffline reco (the vertex is the one reconstructed in HI)
################################### 1st step:pixel-triplet seeding, lower-pT;

from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *

###################################
from RecoTracker.IterativeTracking.LowPtTripletStep_cff import *


# NEW CLUSTERS (remove previously used clusters)
hiRegitLowPtTripletStepClusters = cms.EDProducer("HITrackClusterRemover",
                                     clusterLessSolution= cms.bool(True),
                                     oldClusterRemovalInfo = cms.InputTag("hiRegitInitialStepClusters"),
                                     trajectories = cms.InputTag("hiRegitInitialStepTracks"),
                                     overrideTrkQuals = cms.InputTag('hiRegitInitialStepSelector','hiRegitInitialStep'),
                                     TrackQuality = cms.string('highPurity'),
                                     pixelClusters = cms.InputTag("siPixelClusters"),
                                     stripClusters = cms.InputTag("siStripClusters"),
                                     Common = cms.PSet(
    maxChi2 = cms.double(9.0),
    ),
                                     Strip = cms.PSet(
    maxChi2 = cms.double(9.0),
    #Yen-Jie's mod to preserve merged clusters
    maxSize = cms.uint32(2)
    )
                                     )


# SEEDING LAYERS
hiRegitLowPtTripletStepSeedLayers =  RecoTracker.IterativeTracking.LowPtTripletStep_cff.lowPtTripletStepSeedLayers.clone()
hiRegitLowPtTripletStepSeedLayers.BPix.skipClusters = cms.InputTag('hiRegitLowPtTripletStepClusters')
hiRegitLowPtTripletStepSeedLayers.FPix.skipClusters = cms.InputTag('hiRegitLowPtTripletStepClusters')

# seeds
hiRegitLowPtTripletStepSeeds     = RecoTracker.IterativeTracking.LowPtTripletStep_cff.lowPtTripletStepSeeds.clone()
hiRegitLowPtTripletStepSeeds.RegionFactoryPSet                                           = HiTrackingRegionFactoryFromJetsBlock.clone()
hiRegitLowPtTripletStepSeeds.ClusterCheckPSet.doClusterCheck                             = False # do not check for max number of clusters pixel or strips
hiRegitLowPtTripletStepSeeds.OrderedHitsFactoryPSet.SeedingLayers                                  = 'hiRegitLowPtTripletStepSeedLayers'
from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
import RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi
hiRegitLowPtTripletStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.SeedComparitorPSet = RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi.LowPtClusterShapeSeedComparitor.clone()
hiRegitLowPtTripletStepSeeds.RegionFactoryPSet.RegionPSet.ptMin = 0.4


# building: feed the new-named seeds
hiRegitLowPtTripletStepTrajectoryFilter = RecoTracker.IterativeTracking.LowPtTripletStep_cff.lowPtTripletStepStandardTrajectoryFilter.clone()


hiRegitLowPtTripletStepTrajectoryBuilder = RecoTracker.IterativeTracking.LowPtTripletStep_cff.lowPtTripletStepTrajectoryBuilder.clone(
    trajectoryFilter = cms.PSet(refToPSet_ = cms.string('hiRegitLowPtTripletStepTrajectoryFilter')),
    clustersToSkip = cms.InputTag('hiRegitLowPtTripletStepClusters'),
)

# track candidates
hiRegitLowPtTripletStepTrackCandidates        =  RecoTracker.IterativeTracking.LowPtTripletStep_cff.lowPtTripletStepTrackCandidates.clone(
    src               = cms.InputTag('hiRegitLowPtTripletStepSeeds'),
    TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string('hiRegitLowPtTripletStepTrajectoryBuilder')),
    maxNSeeds = 100000
    )

# fitting: feed new-names
hiRegitLowPtTripletStepTracks                 = RecoTracker.IterativeTracking.LowPtTripletStep_cff.lowPtTripletStepTracks.clone(
    src                 = 'hiRegitLowPtTripletStepTrackCandidates',
    #AlgorithmName = cms.string('pixelLessStep')
    AlgorithmName = cms.string('lowPtTripletStep')
)


# Track selection
import RecoHI.HiTracking.hiMultiTrackSelector_cfi
hiRegitLowPtTripletStepSelector = RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiMultiTrackSelector.clone(
    src='hiRegitLowPtTripletStepTracks',
    trackSelectors= cms.VPSet(
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiLooseMTS.clone(
    name = 'hiRegitLowPtTripletStepLoose',
    d0_par2 = [9999.0, 0.0],
    dz_par2 = [9999.0, 0.0],
    applyAdaptedPVCuts = False
    ), #end of pset
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
    name = 'hiRegitLowPtTripletStepTight',
    preFilterName = 'hiRegitLowPtTripletStepLoose',
    d0_par2 = [9999.0, 0.0],
    dz_par2 = [9999.0, 0.0],
    applyAdaptedPVCuts = False
    ),
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
    name = 'hiRegitLowPtTripletStep',
    preFilterName = 'hiRegitLowPtTripletStepTight',
    d0_par2 = [9999.0, 0.0],
    dz_par2 = [9999.0, 0.0],
    applyAdaptedPVCuts = False
    ),
    ) #end of vpset
    ) #end of clone  


hiRegitLowPtTripletStepTask = cms.Task(hiRegitLowPtTripletStepClusters,
                                       hiRegitLowPtTripletStepSeedLayers,
                                       hiRegitLowPtTripletStepSeeds,
                                       hiRegitLowPtTripletStepTrackCandidates,
                                       hiRegitLowPtTripletStepTracks,
                                       hiRegitLowPtTripletStepSelector)
hiRegitLowPtTripletStep = cms.Sequence(hiRegitLowPtTripletStepTask)


