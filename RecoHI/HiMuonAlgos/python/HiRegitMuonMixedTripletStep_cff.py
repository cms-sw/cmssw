import FWCore.ParameterSet.Config as cms

################################################################################### 
# pp iterative tracking modified for hiOffline reco (the vertex is the one reconstructed in HI)
################################### 4th step: large impact parameter tracking using mixed-triplet seeding

from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.vertexCollection = cms.InputTag("hiSelectedVertex")
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonSrc= cms.InputTag("standAloneMuons","UpdatedAtVtx")

HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.UseVertex      = True
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.UseFixedRegion = True
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.Phi_fixed      = 0.3
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.Eta_fixed      = 0.2


###################################
from RecoTracker.IterativeTracking.MixedTripletStep_cff import *

# NEW CLUSTERS (remove previously used clusters)
hiRegitMuMixedTripletStepClusters = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepClusters.clone(
    oldClusterRemovalInfo = cms.InputTag("hiRegitMuDetachedTripletStepClusters"),
    trajectories          = cms.InputTag("hiRegitMuDetachedTripletStepTracks"),
    overrideTrkQuals      = cms.InputTag('hiRegitMuDetachedTripletStepSelector','hiRegitMuDetachedTripletStep'),
)


# SEEDING LAYERS A
hiRegitMuMixedTripletStepSeedLayersA =  RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSeedLayersA.clone(
    ComponentName = 'hiRegitMuMixedTripletStepSeedLayersA'
    )
hiRegitMuMixedTripletStepSeedLayersA.BPix.skipClusters = cms.InputTag('hiRegitMuMixedTripletStepClusters')
hiRegitMuMixedTripletStepSeedLayersA.FPix.skipClusters = cms.InputTag('hiRegitMuMixedTripletStepClusters')
hiRegitMuMixedTripletStepSeedLayersA.TEC.skipClusters  = cms.InputTag('hiRegitMuMixedTripletStepClusters')

# SEEDS A
hiRegitMuMixedTripletStepSeedsA = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSeedsA.clone()
hiRegitMuMixedTripletStepSeedsA.RegionFactoryPSet                                           = HiTrackingRegionFactoryFromSTAMuonsBlock.clone()
hiRegitMuMixedTripletStepSeedsA.ClusterCheckPSet.doClusterCheck                             = False # do not check for max number of clusters pixel or strips
hiRegitMuMixedTripletStepSeedsA.RegionFactoryPSet.MuonTrackingRegionBuilder.EscapePt        = 1.3
hiRegitMuMixedTripletStepSeedsA.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaR          = 0.5 # default = 0.2
hiRegitMuMixedTripletStepSeedsA.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaZ_Region   = 0.5 # this give you the length 
hiRegitMuMixedTripletStepSeedsA.RegionFactoryPSet.MuonTrackingRegionBuilder.Rescale_Dz      = 4.   # max(DeltaZ_Region,Rescale_Dz*vtx->zError())
hiRegitMuMixedTripletStepSeedsA.OrderedHitsFactoryPSet.SeedingLayers = 'hiRegitMuMixedTripletStepSeedLayersA'

# SEEDING LAYERS B
hiRegitMuMixedTripletStepSeedLayersB =  RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSeedLayersB.clone(
    ComponentName = 'hiRegitMuMixedTripletStepSeedLayersB',
    )
hiRegitMuMixedTripletStepSeedLayersB.BPix.skipClusters = cms.InputTag('hiRegitMuMixedTripletStepClusters')
hiRegitMuMixedTripletStepSeedLayersB.TIB.skipClusters  = cms.InputTag('hiRegitMuMixedTripletStepClusters')


hiRegitMuMixedTripletStepSeedsB = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSeedsB.clone()
hiRegitMuMixedTripletStepSeedsB.RegionFactoryPSet                                           = HiTrackingRegionFactoryFromSTAMuonsBlock.clone()
hiRegitMuMixedTripletStepSeedsB.ClusterCheckPSet.doClusterCheck                             = False # do not check for max number of clusters pixel or strips
hiRegitMuMixedTripletStepSeedsB.RegionFactoryPSet.MuonTrackingRegionBuilder.EscapePt        = 1.5
hiRegitMuMixedTripletStepSeedsB.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaR          = 0.5 # default = 0.2
hiRegitMuMixedTripletStepSeedsB.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaZ_Region   = 0.5 # this give you the length 
hiRegitMuMixedTripletStepSeedsB.RegionFactoryPSet.MuonTrackingRegionBuilder.Rescale_Dz      = 4.   # max(DeltaZ_Region,Rescale_Dz*vtx->zError())
hiRegitMuMixedTripletStepSeedsB.OrderedHitsFactoryPSet.SeedingLayers = 'hiRegitMuMixedTripletStepSeedLayersB'

# combine seeds
hiRegitMuMixedTripletStepSeeds = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSeeds.clone(
    seedCollections = cms.VInputTag(
        cms.InputTag('hiRegitMuMixedTripletStepSeedsA'),
        cms.InputTag('hiRegitMuMixedTripletStepSeedsB'),
        )
    )

# track building
hiRegitMuMixedTripletStepTrajectoryFilter = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepTrajectoryFilter.clone(
    ComponentName   = 'hiRegitMuMixedTripletStepTrajectoryFilter',
    )

hiRegitMuMixedTripletStepTrajectoryFilter.filterPset.minPt = 1.
hiRegitMuMixedTripletStepTrajectoryFilter.filterPset.minimumNumberOfHits = 6
hiRegitMuMixedTripletStepTrajectoryFilter.filterPset.minHitsMinPt        = 4


 # after each new hit, apply pT cut for traj w/ at least minHitsMinPt = cms.int32(3),

hiRegitMuMixedTripletStepTrajectoryBuilder = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepTrajectoryBuilder.clone(
    ComponentName        = 'hiRegitMuMixedTripletStepTrajectoryBuilder',
    trajectoryFilterName = 'hiRegitMuMixedTripletStepTrajectoryFilter',
    clustersToSkip       = cms.InputTag('hiRegitMuMixedTripletStepClusters'),
    minNrOfHitsForRebuild = 6 #change from default 4
)

hiRegitMuMixedTripletStepTrackCandidates        =  RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepTrackCandidates.clone(
    src               = cms.InputTag('hiRegitMuMixedTripletStepSeeds'),
    TrajectoryBuilder = 'hiRegitMuMixedTripletStepTrajectoryBuilder',
    maxNSeeds         = cms.uint32(1000000)
    )

# fitting: feed new-names
hiRegitMuMixedTripletStepTracks                 = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepTracks.clone(
    src                 = 'hiRegitMuMixedTripletStepTrackCandidates'
)


# TRACK SELECTION AND QUALITY FLAG SETTING.
#import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
from RecoHI.HiTracking.hiRegitMixedTripletStep_cff import *
hiRegitMuMixedTripletStepSelector =  RecoHI.HiTracking.hiRegitMixedTripletStep_cff.hiRegitMixedTripletStepSelector.clone( # selector from hi taken
    src                 = 'hiRegitMuMixedTripletStepTracks',
    trackSelectors= cms.VPSet(
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiLooseMTS.clone(
            name = 'hiRegitMuMixedTripletStepLoose',
            ),
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
            name = 'hiRegitMuMixedTripletStepTight',
            preFilterName = 'hiRegitMuMixedTripletStepLoose',
            ),
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
            name = 'hiRegitMuMixedTripletStep',
            preFilterName = 'hiRegitMuMixedTripletStepTight',
            )
        ) #end of vpset
    ) #end of clone

hiRegitMuonMixedTripletStep = cms.Sequence(hiRegitMuMixedTripletStepClusters*
                                         hiRegitMuMixedTripletStepSeedsA*
                                         hiRegitMuMixedTripletStepSeedsB*
                                         hiRegitMuMixedTripletStepSeeds*
                                         hiRegitMuMixedTripletStepTrackCandidates*
                                         hiRegitMuMixedTripletStepTracks*
                                         hiRegitMuMixedTripletStepSelector)
