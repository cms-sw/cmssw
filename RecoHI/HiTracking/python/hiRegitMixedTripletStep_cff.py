import FWCore.ParameterSet.Config as cms

################################################################################### 
# pp iterative tracking modified for hiOffline reco (the vertex is the one reconstructed in HI)
################################### 4th step: large impact parameter tracking using mixed-triplet seeding

from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *

###################################
from RecoTracker.IterativeTracking.MixedTripletStep_cff import *

# NEW CLUSTERS (remove previously used clusters)
hiRegitMixedTripletStepClusters = cms.EDProducer("TrackClusterRemover",
                                                clusterLessSolution= cms.bool(True),
                                                oldClusterRemovalInfo = cms.InputTag("hiRegitDetachedTripletStepClusters"),
                                                trajectories = cms.InputTag("hiRegitDetachedTripletStepTracks"),
                                                overrideTrkQuals = cms.InputTag('hiRegitDetachedTripletStepSelector','hiRegitDetachedTripletStep'),
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



# SEEDING LAYERS A
hiRegitMixedTripletStepSeedLayersA =  RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSeedLayersA.clone(
    ComponentName = 'hiRegitMixedTripletStepSeedLayersA'
    )
hiRegitMixedTripletStepSeedLayersA.BPix.skipClusters = cms.InputTag('hiRegitMixedTripletStepClusters')
hiRegitMixedTripletStepSeedLayersA.FPix.skipClusters = cms.InputTag('hiRegitMixedTripletStepClusters')
hiRegitMixedTripletStepSeedLayersA.TEC.skipClusters  = cms.InputTag('hiRegitMixedTripletStepClusters')

# SEEDS A
hiRegitMixedTripletStepSeedsA = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSeedsA.clone()
hiRegitMixedTripletStepSeedsA.RegionFactoryPSet                                           = HiTrackingRegionFactoryFromJetsBlock.clone()
hiRegitMixedTripletStepSeedsA.ClusterCheckPSet.doClusterCheck                             = False # do not check for max number of clusters pixel or strips
hiRegitMixedTripletStepSeedsA.OrderedHitsFactoryPSet.SeedingLayers = 'hiRegitMixedTripletStepSeedLayersA'

# SEEDING LAYERS B
hiRegitMixedTripletStepSeedLayersB =  RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSeedLayersB.clone(
    ComponentName = 'hiRegitMixedTripletStepSeedLayersB',
    )
hiRegitMixedTripletStepSeedLayersB.BPix.skipClusters = cms.InputTag('hiRegitMixedTripletStepClusters')
hiRegitMixedTripletStepSeedLayersB.TIB.skipClusters  = cms.InputTag('hiRegitMixedTripletStepClusters')


hiRegitMixedTripletStepSeedsB = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSeedsB.clone()
hiRegitMixedTripletStepSeedsB.RegionFactoryPSet                                           = HiTrackingRegionFactoryFromJetsBlock.clone()
hiRegitMixedTripletStepSeedsB.ClusterCheckPSet.doClusterCheck                             = False # do not check for max number of clusters pixel or strips
hiRegitMixedTripletStepSeedsB.OrderedHitsFactoryPSet.SeedingLayers = 'hiRegitMixedTripletStepSeedLayersB'

# combine seeds
hiRegitMixedTripletStepSeeds = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSeeds.clone(
    seedCollections = cms.VInputTag(
        cms.InputTag('hiRegitMixedTripletStepSeedsA'),
        cms.InputTag('hiRegitMixedTripletStepSeedsB'),
        )
    )

# track building
hiRegitMixedTripletStepTrajectoryFilter = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepTrajectoryFilter.clone(
    ComponentName        = 'hiRegitMixedTripletStepTrajectoryFilter'
    )
#hiRegitMixedTripletStepTrajectoryFilter.filterPset.minPt     = 1.15 # after each new hit, apply pT cut for traj w/ at least minHitsMinPt = cms.int32(3),

hiRegitMixedTripletStepTrajectoryBuilder = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepTrajectoryBuilder.clone(
    ComponentName        = 'hiRegitMixedTripletStepTrajectoryBuilder',
    trajectoryFilterName = 'hiRegitMixedTripletStepTrajectoryFilter',
    clustersToSkip       = cms.InputTag('hiRegitMixedTripletStepClusters'),
)

hiRegitMixedTripletStepTrackCandidates        =  RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepTrackCandidates.clone(
    src               = cms.InputTag('hiRegitMixedTripletStepSeeds'),
    TrajectoryBuilder = 'hiRegitMixedTripletStepTrajectoryBuilder'
    )

# fitting: feed new-names
hiRegitMixedTripletStepTracks                 = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepTracks.clone(
    src                 = 'hiRegitMixedTripletStepTrackCandidates',
    #AlgorithmName = cms.string('iter8'),
    AlgorithmName = cms.string('iter4'),
    )

# Track selection
import RecoHI.HiTracking.hiMultiTrackSelector_cfi
hiRegitMixedTripletStepSelector = RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiMultiTrackSelector.clone(
    src='hiRegitMixedTripletStepTracks',
    trackSelectors= cms.VPSet(
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiLooseMTS.clone(
    name = 'hiRegitMixedTripletStepLoose',
    d0_par2 = [9999.0, 0.0],
    dz_par2 = [9999.0, 0.0],
    applyAdaptedPVCuts = False
    ), #end of pset
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
    name = 'hiRegitMixedTripletStepTight',
    preFilterName = 'hiRegitMixedTripletStepLoose',
    d0_par2 = [9999.0, 0.0],
    dz_par2 = [9999.0, 0.0],
    applyAdaptedPVCuts = False
    ),
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
    name = 'hiRegitMixedTripletStep',
    preFilterName = 'hiRegitMixedTripletStepTight',
    d0_par2 = [9999.0, 0.0],
    dz_par2 = [9999.0, 0.0],
    applyAdaptedPVCuts = False
    ),
    ) #end of vpset
    ) #end of clone  

hiRegitMixedTripletStep = cms.Sequence(hiRegitMixedTripletStepClusters*
                                       hiRegitMixedTripletStepSeedsA*
                                       hiRegitMixedTripletStepSeedsB*
                                       hiRegitMixedTripletStepSeeds*
                                       hiRegitMixedTripletStepTrackCandidates*
                                       hiRegitMixedTripletStepTracks*
                                       hiRegitMixedTripletStepSelector
                                       )
