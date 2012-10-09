import FWCore.ParameterSet.Config as cms

################################################################################### 
# pp iterative tracking modified for hiOffline reco (the vertex is the one reconstructed in HI)
################################### 3rd step: low-pT and displaced tracks from pixel triplets

from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.vertexCollection = cms.InputTag("hiSelectedVertex")
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonSrc= cms.InputTag("standAloneMuons","UpdatedAtVtx")

HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.UseVertex      = True

HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.UseFixedRegion = True
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.Phi_fixed      = 0.2
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.Eta_fixed      = 0.2


###################################
from RecoTracker.IterativeTracking.DetachedTripletStep_cff import *

# NEW CLUSTERS (remove previously used clusters)
hiRegitMuDetachedTripletStepClusters = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepClusters.clone(
    oldClusterRemovalInfo = cms.InputTag("hiRegitMuPixelPairStepClusters"),
    trajectories          = cms.InputTag("hiRegitMuPixelPairStepTracks"),
    overrideTrkQuals      = cms.InputTag('hiRegitMuPixelPairStepSelector','hiRegitMuPixelPairStep'),
)


# SEEDING LAYERS
hiRegitMuDetachedTripletStepSeedLayers =  RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepSeedLayers.clone(
    ComponentName = 'hiRegitMuDetachedTripletStepSeedLayers'
    )
hiRegitMuDetachedTripletStepSeedLayers.BPix.skipClusters = cms.InputTag('hiRegitMuDetachedTripletStepClusters')
hiRegitMuDetachedTripletStepSeedLayers.FPix.skipClusters = cms.InputTag('hiRegitMuDetachedTripletStepClusters')

# seeding
hiRegitMuDetachedTripletStepSeeds     = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepSeeds.clone()
hiRegitMuDetachedTripletStepSeeds.RegionFactoryPSet                                           = HiTrackingRegionFactoryFromSTAMuonsBlock.clone()
hiRegitMuDetachedTripletStepSeeds.ClusterCheckPSet.doClusterCheck                             = False # do not check for max number of clusters pixel or strips
hiRegitMuDetachedTripletStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.EscapePt        = 1.0
hiRegitMuDetachedTripletStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaR          = 1.0 # default = 0.2
hiRegitMuDetachedTripletStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaZ_Region   = 0.0 # this give you the length 
hiRegitMuDetachedTripletStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.Rescale_Dz      = 4. # max(DeltaZ_Region,Rescale_Dz*vtx->zError())
hiRegitMuDetachedTripletStepSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'hiRegitMuDetachedTripletStepSeedLayers'
from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
hiRegitMuDetachedTripletStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.SeedComparitorPSet.ComponentName = 'LowPtClusterShapeSeedComparitor'

# building: feed the new-named seeds
hiRegitMuDetachedTripletStepTrajectoryFilter = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepTrajectoryFilter.clone(
    ComponentName    = 'hiRegitMuDetachedTripletStepTrajectoryFilter'
    )
hiRegitMuDetachedTripletStepTrajectoryFilter.filterPset.minPt = 1. # after each new hit, apply pT cut for traj w/ at least minHitsMinPt = cms.int32(3),

hiRegitMuDetachedTripletStepTrajectoryBuilder = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepTrajectoryBuilder.clone(
    ComponentName        = 'hiRegitMuDetachedTripletStepTrajectoryBuilder',
    trajectoryFilterName = 'hiRegitMuDetachedTripletStepTrajectoryFilter',
    clustersToSkip       = cms.InputTag('hiRegitMuDetachedTripletStepClusters')
)

hiRegitMuDetachedTripletStepTrackCandidates        =  RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepTrackCandidates.clone(
    src               = cms.InputTag('hiRegitMuDetachedTripletStepSeeds'),
    TrajectoryBuilder = 'hiRegitMuDetachedTripletStepTrajectoryBuilder'
    )

# fitting: feed new-names
hiRegitMuDetachedTripletStepTracks                 = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepTracks.clone(
    src                 = 'hiRegitMuDetachedTripletStepTrackCandidates'
)


hiRegitMuDetachedTripletStepSelector               = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepSelector.clone( 
    src                 ='hiRegitMuDetachedTripletStepTracks',
    vertices            = cms.InputTag("hiSelectedVertex"),
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'hiRegitMuDetachedTripletStepVtxLoose',
            chi2n_par = 1.6,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            d0_par1 = ( 1.2, 3.0 ),
            dz_par1 = ( 1.2, 3.0 ),
            d0_par2 = ( 1.3, 3.0 ),
            dz_par2 = ( 1.3, 3.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'hiRegitMuDetachedTripletStepTrkLoose',
            chi2n_par = 0.7,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            d0_par1 = ( 1.6, 4.0 ),
            dz_par1 = ( 1.6, 4.0 ),
            d0_par2 = ( 1.6, 4.0 ),
            dz_par2 = ( 1.6, 4.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'hiRegitMuDetachedTripletStepVtxTight',
            preFilterName = 'hiRegitMuDetachedTripletStepVtxLoose',
            chi2n_par = 0.7,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.95, 3.0 ),
            dz_par1 = ( 0.9, 3.0 ),
            d0_par2 = ( 1.0, 3.0 ),
            dz_par2 = ( 1.0, 3.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'hiRegitMuDetachedTripletStepTrkTight',
            preFilterName = 'hiRegitMuDetachedTripletStepTrkLoose',
            chi2n_par = 0.5,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 5,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 3,
            d0_par1 = ( 1.1, 4.0 ),
            dz_par1 = ( 1.1, 4.0 ),
            d0_par2 = ( 1.1, 4.0 ),
            dz_par2 = ( 1.1, 4.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'hiRegitMuDetachedTripletStepVtx',
            preFilterName = 'hiRegitMuDetachedTripletStepVtxTight',
            chi2n_par = 0.7,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.85, 3.0 ),
            dz_par1 = ( 0.8, 3.0 ),
            d0_par2 = ( 0.9, 3.0 ),
            dz_par2 = ( 0.9, 3.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'hiRegitMuDetachedTripletStepTrk',
            preFilterName = 'hiRegitMuDetachedTripletStepTrkTight',
            chi2n_par = 0.4,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 5,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 3,
            d0_par1 = ( 1.0, 4.0 ),
            dz_par1 = ( 1.0, 4.0 ),
            d0_par2 = ( 1.0, 4.0 ),
            dz_par2 = ( 1.0, 4.0 )
            )
        ) #end of vpset
    )

hiRegitMuDetachedTripletStep = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStep.clone(
    TrackProducers     = cms.VInputTag(cms.InputTag('hiRegitMuDetachedTripletStepTracks'),
                                       cms.InputTag('hiRegitMuDetachedTripletStepTracks')),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("hiRegitMuDetachedTripletStepSelector","hiRegitMuDetachedTripletStepVtx"),
                                       cms.InputTag("hiRegitMuDetachedTripletStepSelector","hiRegitMuDetachedTripletStepTrk"))
)
    

hiRegitMuonDetachedTripletStep = cms.Sequence(hiRegitMuDetachedTripletStepClusters*
                                              hiRegitMuDetachedTripletStepSeeds*
                                              hiRegitMuDetachedTripletStepTrackCandidates*
                                              hiRegitMuDetachedTripletStepTracks*
                                              hiRegitMuDetachedTripletStepSelector*
                                              hiRegitMuDetachedTripletStep
                                            )

