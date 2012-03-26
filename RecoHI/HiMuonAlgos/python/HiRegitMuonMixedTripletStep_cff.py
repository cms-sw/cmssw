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
from RecoHI.HiTracking.hiRegitMixedTripletStep_cff import *

# NEW CLUSTERS (remove previously used clusters)
hiRegitMuMixedTripletStepClusters = RecoHI.HiTracking.hiRegitMixedTripletStep_cff.hiRegitMixedTripletStepClusters.clone(
    oldClusterRemovalInfo = cms.InputTag("hiRegitMuDetachedTripletStepClusters"),
    trajectories          = cms.InputTag("hiRegitMuDetachedTripletStepTracks"),
    overrideTrkQuals      = cms.InputTag('hiRegitMuDetachedTripletStepSelector','hiRegitMuDetachedTripletStep'),
)


# SEEDING LAYERS A
hiRegitMuMixedTripletStepSeedLayersA =  RecoHI.HiTracking.hiRegitMixedTripletStep_cff.hiRegitMixedTripletStepSeedLayersA.clone(
    ComponentName = 'hiRegitMuMixedTripletStepSeedLayersA'
    )
hiRegitMuMixedTripletStepSeedLayersA.BPix.skipClusters = cms.InputTag('hiRegitMuMixedTripletStepClusters')
hiRegitMuMixedTripletStepSeedLayersA.FPix.skipClusters = cms.InputTag('hiRegitMuMixedTripletStepClusters')
hiRegitMuMixedTripletStepSeedLayersA.TEC.skipClusters  = cms.InputTag('hiRegitMuMixedTripletStepClusters')

# SEEDS A
hiRegitMuMixedTripletStepSeedsA = RecoHI.HiTracking.hiRegitMixedTripletStep_cff.hiRegitMixedTripletStepSeedsA.clone()
hiRegitMuMixedTripletStepSeedsA.RegionFactoryPSet                                           = HiTrackingRegionFactoryFromSTAMuonsBlock.clone()
hiRegitMuMixedTripletStepSeedsA.ClusterCheckPSet.doClusterCheck                             = False # do not check for max number of clusters pixel or strips
hiRegitMuMixedTripletStepSeedsA.RegionFactoryPSet.MuonTrackingRegionBuilder.EscapePt        = 1.0
hiRegitMuMixedTripletStepSeedsA.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaR          = 2.0 # default = 0.2
hiRegitMuMixedTripletStepSeedsA.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaZ_Region   = 2.0 # this give you the length 
hiRegitMuMixedTripletStepSeedsA.RegionFactoryPSet.MuonTrackingRegionBuilder.Rescale_Dz      = 20.   # max(DeltaZ_Region,Rescale_Dz*vtx->zError())
hiRegitMuMixedTripletStepSeedsA.OrderedHitsFactoryPSet.SeedingLayers = 'hiRegitMuMixedTripletStepSeedLayersA'

# SEEDING LAYERS B
hiRegitMuMixedTripletStepSeedLayersB =  RecoHI.HiTracking.hiRegitMixedTripletStep_cff.hiRegitMixedTripletStepSeedLayersB.clone(
    ComponentName = 'hiRegitMuMixedTripletStepSeedLayersB',
    )
hiRegitMuMixedTripletStepSeedLayersB.BPix.skipClusters = cms.InputTag('hiRegitMuMixedTripletStepClusters')
hiRegitMuMixedTripletStepSeedLayersB.TIB.skipClusters  = cms.InputTag('hiRegitMuMixedTripletStepClusters')


hiRegitMuMixedTripletStepSeedsB = RecoHI.HiTracking.hiRegitMixedTripletStep_cff.hiRegitMixedTripletStepSeedsB.clone()
hiRegitMuMixedTripletStepSeedsB.RegionFactoryPSet                                           = HiTrackingRegionFactoryFromSTAMuonsBlock.clone()
hiRegitMuMixedTripletStepSeedsB.ClusterCheckPSet.doClusterCheck                             = False # do not check for max number of clusters pixel or strips
hiRegitMuMixedTripletStepSeedsB.RegionFactoryPSet.MuonTrackingRegionBuilder.EscapePt        = 1.5
hiRegitMuMixedTripletStepSeedsB.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaR          = 2.0 # default = 0.2
hiRegitMuMixedTripletStepSeedsB.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaZ_Region   = 2.0 # this give you the length 
hiRegitMuMixedTripletStepSeedsB.RegionFactoryPSet.MuonTrackingRegionBuilder.Rescale_Dz      = 20.   # max(DeltaZ_Region,Rescale_Dz*vtx->zError())
hiRegitMuMixedTripletStepSeedsB.OrderedHitsFactoryPSet.SeedingLayers = 'hiRegitMuMixedTripletStepSeedLayersB'

# combine seeds
hiRegitMuMixedTripletStepSeeds = RecoHI.HiTracking.hiRegitMixedTripletStep_cff.hiRegitMixedTripletStepSeeds.clone(
    seedCollections = cms.VInputTag(
        cms.InputTag('hiRegitMuMixedTripletStepSeedsA'),
        cms.InputTag('hiRegitMuMixedTripletStepSeedsB'),
        )
    )

# track building
hiRegitMuMixedTripletStepTrajectoryFilter = RecoHI.HiTracking.hiRegitMixedTripletStep_cff.hiRegitMixedTripletStepTrajectoryFilter.clone(
    ComponentName        = 'hiRegitMuMixedTripletStepTrajectoryFilter'
    )
hiRegitMuMixedTripletStepTrajectoryFilter.filterPset.minPt           = 0.9


 # after each new hit, apply pT cut for traj w/ at least minHitsMinPt = cms.int32(3),

hiRegitMuMixedTripletStepTrajectoryBuilder = RecoHI.HiTracking.hiRegitMixedTripletStep_cff.hiRegitMixedTripletStepTrajectoryBuilder.clone(
    ComponentName        = 'hiRegitMuMixedTripletStepTrajectoryBuilder',
    trajectoryFilterName = 'hiRegitMuMixedTripletStepTrajectoryFilter',
    clustersToSkip       = cms.InputTag('hiRegitMuMixedTripletStepClusters'),
)

hiRegitMuMixedTripletStepTrackCandidates        =  RecoHI.HiTracking.hiRegitMixedTripletStep_cff.hiRegitMixedTripletStepTrackCandidates.clone(
    src               = cms.InputTag('hiRegitMuMixedTripletStepSeeds'),
    TrajectoryBuilder = 'hiRegitMuMixedTripletStepTrajectoryBuilder',
    maxNSeeds         = cms.uint32(1000000)
    )

# fitting: feed new-names
hiRegitMuMixedTripletStepTracks                 = RecoHI.HiTracking.hiRegitMixedTripletStep_cff.hiRegitMixedTripletStepTracks.clone(
    src                 = 'hiRegitMuMixedTripletStepTrackCandidates'
)


# TRACK SELECTION AND QUALITY FLAG SETTING.
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
hiRegitMuMixedTripletStepSelector =  RecoHI.HiTracking.hiRegitMixedTripletStep_cff.hiRegitMixedTripletStepSelector.clone(
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
