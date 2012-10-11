import FWCore.ParameterSet.Config as cms

################################################################################### 
# pp iterative tracking modified for hiOffline reco (the vertex is the one reconstructed in HI)
################################### 4th step: large impact parameter tracking using mixed-triplet seeding

from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.vertexCollection = cms.InputTag("hiSelectedVertex")
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonSrc= cms.InputTag("standAloneMuons","UpdatedAtVtx")

HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.UseVertex      = True

HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.UseFixedRegion = True
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.Phi_fixed      = 0.2
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.Eta_fixed      = 0.2


###################################
from RecoTracker.IterativeTracking.MixedTripletStep_cff import *

# NEW CLUSTERS (remove previously used clusters)
hiRegitMuMixedTripletStepClusters = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepClusters.clone(
    oldClusterRemovalInfo = cms.InputTag("hiRegitMuDetachedTripletStepClusters"),
    trajectories          = cms.InputTag("hiRegitMuDetachedTripletStepTracks"),
    overrideTrkQuals      = cms.InputTag('hiRegitMuDetachedTripletStep'),
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
hiRegitMuMixedTripletStepSeedsA.RegionFactoryPSet.MuonTrackingRegionBuilder.EscapePt        = 1.35
hiRegitMuMixedTripletStepSeedsA.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaR          = 2.0 # default = 0.2
hiRegitMuMixedTripletStepSeedsA.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaZ_Region   = 10. # this give you the length 
hiRegitMuMixedTripletStepSeedsA.RegionFactoryPSet.MuonTrackingRegionBuilder.Rescale_Dz      = 0.   # max(DeltaZ_Region,Rescale_Dz*vtx->zError())
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
hiRegitMuMixedTripletStepSeedsB.RegionFactoryPSet.MuonTrackingRegionBuilder.EscapePt        = 1.50
hiRegitMuMixedTripletStepSeedsB.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaR          = 2.0 # default = 0.2
hiRegitMuMixedTripletStepSeedsB.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaZ_Region   = 10. # this give you the length 
hiRegitMuMixedTripletStepSeedsB.RegionFactoryPSet.MuonTrackingRegionBuilder.Rescale_Dz      = 0.   # max(DeltaZ_Region,Rescale_Dz*vtx->zError())
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
    ComponentName        = 'hiRegitMuMixedTripletStepTrajectoryFilter'
    )
hiRegitMuMixedTripletStepTrajectoryFilter.filterPset.minPt     = 1.15 # after each new hit, apply pT cut for traj w/ at least minHitsMinPt = cms.int32(3),

hiRegitMuMixedTripletStepTrajectoryBuilder = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepTrajectoryBuilder.clone(
    ComponentName        = 'hiRegitMuMixedTripletStepTrajectoryBuilder',
    trajectoryFilterName = 'hiRegitMuMixedTripletStepTrajectoryFilter',
    clustersToSkip       = cms.InputTag('hiRegitMuMixedTripletStepClusters'),
)

hiRegitMuMixedTripletStepTrackCandidates        =  RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepTrackCandidates.clone(
    src               = cms.InputTag('hiRegitMuMixedTripletStepSeeds'),
    TrajectoryBuilder = 'hiRegitMuMixedTripletStepTrajectoryBuilder'
    )

# fitting: feed new-names
hiRegitMuMixedTripletStepTracks                 = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepTracks.clone(
    src                 = 'hiRegitMuMixedTripletStepTrackCandidates'
)


# TRACK SELECTION AND QUALITY FLAG SETTING.
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
hiRegitMuMixedTripletStepSelector =  RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSelector.clone(
    src                 = 'hiRegitMuMixedTripletStepTracks',
    vertices            = cms.InputTag("hiSelectedVertex"),
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'hiRegitMuMixedTripletStepVtxLoose',
            chi2n_par = 1.2,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 2,
            d0_par1 = ( 1.2, 3.0 ),
            dz_par1 = ( 1.2, 3.0 ),
            d0_par2 = ( 1.3, 3.0 ),
            dz_par2 = ( 1.3, 3.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'hiRegitMuMixedTripletStepTrkLoose',
            chi2n_par = 0.6,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 4,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 3,
            d0_par1 = ( 1.2, 4.0 ),
            dz_par1 = ( 1.2, 4.0 ),
            d0_par2 = ( 1.2, 4.0 ),
            dz_par2 = ( 1.2, 4.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'hiRegitMuMixedTripletStepVtxTight',
            preFilterName = 'hiRegitMuMixedTripletStepVtxLoose',
            chi2n_par = 0.6,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 3,
            d0_par1 = ( 1.1, 3.0 ),
            dz_par1 = ( 1.1, 3.0 ),
            d0_par2 = ( 1.2, 3.0 ),
            dz_par2 = ( 1.2, 3.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'hiRegitMuMixedTripletStepTrkTight',
            preFilterName = 'hiRegitMuMixedTripletStepTrkLoose',
            chi2n_par = 0.4,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 5,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 4,
            d0_par1 = ( 1.1, 4.0 ),
            dz_par1 = ( 1.1, 4.0 ),
            d0_par2 = ( 1.1, 4.0 ),
            dz_par2 = ( 1.1, 4.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'hiRegitMuMixedTripletStepVtx',
            preFilterName = 'hiRegitMuMixedTripletStepVtxTight',
            chi2n_par = 0.4,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 3,
            d0_par1 = ( 1.1, 3.0 ),
            dz_par1 = ( 1.1, 3.0 ),
            d0_par2 = ( 1.2, 3.0 ),
            dz_par2 = ( 1.2, 3.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'hiRegitMuMixedTripletStepTrk',
            preFilterName = 'hiRegitMuMixedTripletStepTrkTight',
            chi2n_par = 0.3,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 5,
            maxNumberLostLayers = 0,
            minNumber3DLayers = 4,
            d0_par1 = ( 0.9, 4.0 ),
            dz_par1 = ( 0.9, 4.0 ),
            d0_par2 = ( 0.9, 4.0 ),
            dz_par2 = ( 0.9, 4.0 )
            )
        ) #end of vpset
    ) #end of clone


hiRegitMuMixedTripletStep = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStep.clone(
    TrackProducers = cms.VInputTag(cms.InputTag('hiRegitMuMixedTripletStepTracks'),
                                   cms.InputTag('hiRegitMuMixedTripletStepTracks')),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("hiRegitMuMixedTripletStepSelector","hiRegitMuMixedTripletStepVtx"),
                                       cms.InputTag("hiRegitMuMixedTripletStepSelector","hiRegitMuMixedTripletStepTrk"))
    )

hiRegitMuonMixedTripletStep = cms.Sequence(hiRegitMuMixedTripletStepClusters*
                                         hiRegitMuMixedTripletStepSeedsA*
                                         hiRegitMuMixedTripletStepSeedsB*
                                         hiRegitMuMixedTripletStepSeeds*
                                         hiRegitMuMixedTripletStepTrackCandidates*
                                         hiRegitMuMixedTripletStepTracks*
                                         hiRegitMuMixedTripletStepSelector*
                                         hiRegitMuMixedTripletStep)
