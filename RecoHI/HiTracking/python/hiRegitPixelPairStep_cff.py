import FWCore.ParameterSet.Config as cms

################################################################################### 
# pp iterative tracking modified for hiOffline reco (the vertex is the one reconstructed in HI)
################################### 2nd step: pixel pairs

from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *

###################################
from RecoTracker.IterativeTracking.PixelPairStep_cff import *

# NEW CLUSTERS (remove previously used clusters)
hiRegitPixelPairStepClusters = cms.EDProducer("HITrackClusterRemover",
                                     clusterLessSolution= cms.bool(True),
                                     oldClusterRemovalInfo = cms.InputTag("hiRegitLowPtTripletStepClusters"),
                                     trajectories = cms.InputTag("hiRegitLowPtTripletStepTracks"),
                                     overrideTrkQuals = cms.InputTag('hiRegitLowPtTripletStepSelector','hiRegitLowPtTripletStep'),
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
hiRegitPixelPairStepSeedLayers =  RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepSeedLayers.clone()
hiRegitPixelPairStepSeedLayers.BPix.skipClusters = cms.InputTag('hiRegitPixelPairStepClusters')
hiRegitPixelPairStepSeedLayers.FPix.skipClusters = cms.InputTag('hiRegitPixelPairStepClusters')



# seeding
hiRegitPixelPairStepSeeds     = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepSeeds.clone()
hiRegitPixelPairStepSeeds.RegionFactoryPSet                                           = HiTrackingRegionFactoryFromJetsBlock.clone()
hiRegitPixelPairStepSeeds.ClusterCheckPSet.doClusterCheck                             = False # do not check for max number of clusters pixel or strips
hiRegitPixelPairStepSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'hiRegitPixelPairStepSeedLayers'
hiRegitPixelPairStepSeeds.RegionFactoryPSet.RegionPSet.ptMin = 1.2


# building: feed the new-named seeds
hiRegitPixelPairStepTrajectoryFilter = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepTrajectoryFilterBase.clone()

hiRegitPixelPairStepTrajectoryBuilder = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepTrajectoryBuilder.clone(
    trajectoryFilter     = cms.PSet(refToPSet_ = cms.string('hiRegitPixelPairStepTrajectoryFilter')),
    clustersToSkip       = cms.InputTag('hiRegitPixelPairStepClusters'),
)

# trackign candidate
hiRegitPixelPairStepTrackCandidates        =  RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepTrackCandidates.clone(
    src               = cms.InputTag('hiRegitPixelPairStepSeeds'),
    TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string('hiRegitPixelPairStepTrajectoryBuilder')),
    maxNSeeds = 100000
    )

# fitting: feed new-names
hiRegitPixelPairStepTracks                 = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepTracks.clone(
    src                 = 'hiRegitPixelPairStepTrackCandidates',
    AlgorithmName = cms.string('pixelPairStep'),
    )


# Track selection
import RecoHI.HiTracking.hiMultiTrackSelector_cfi
hiRegitPixelPairStepSelector = RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiMultiTrackSelector.clone(
    src='hiRegitPixelPairStepTracks',
    trackSelectors= cms.VPSet(
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiLooseMTS.clone(
    name = 'hiRegitPixelPairStepLoose',
    d0_par2 = [9999.0, 0.0],
    dz_par2 = [9999.0, 0.0],
    applyAdaptedPVCuts = False
    ), #end of pset
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
    name = 'hiRegitPixelPairStepTight',
    preFilterName = 'hiRegitPixelPairStepLoose',
    d0_par2 = [9999.0, 0.0],
    dz_par2 = [9999.0, 0.0],
    applyAdaptedPVCuts = False
    ),
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
    name = 'hiRegitPixelPairStep',
    preFilterName = 'hiRegitPixelPairStepTight',
    d0_par2 = [9999.0, 0.0],
    dz_par2 = [9999.0, 0.0],
    applyAdaptedPVCuts = False
    ),
    ) #end of vpset
    ) #end of clone  

hiRegitPixelPairStepTask = cms.Task(hiRegitPixelPairStepClusters,
                                    hiRegitPixelPairStepSeedLayers,
                                    hiRegitPixelPairStepSeeds,
                                    hiRegitPixelPairStepTrackCandidates,
                                    hiRegitPixelPairStepTracks,
                                    hiRegitPixelPairStepSelector)
hiRegitPixelPairStep = cms.Sequence(hiRegitPixelPairStepTask)
