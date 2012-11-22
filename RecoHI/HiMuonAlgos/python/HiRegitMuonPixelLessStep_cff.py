import FWCore.ParameterSet.Config as cms

# pp iterative tracking modified for hiOffline reco (the vertex is the one reconstructed in HI)
################################### 5th step: large impact parameter tracking using TIB/TID/TEC stereo layer seeding

from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.vertexCollection = cms.InputTag("hiSelectedVertex")
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonSrc= cms.InputTag("standAloneMuons","UpdatedAtVtx")

HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.UseVertex      = True

HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.UseFixedRegion = True
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.Phi_fixed      = 0.2
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.Eta_fixed      = 0.2

###################################
from RecoTracker.IterativeTracking.PixelLessStep_cff import *

# remove previously used clusters
hiRegitMuPixelLessStepClusters = RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepClusters.clone(
    oldClusterRemovalInfo = cms.InputTag("hiRegitMuMixedTripletStepClusters"),
    trajectories     = cms.InputTag("hiRegitMuMixedTripletStepTracks"),
    overrideTrkQuals = cms.InputTag('hiRegitMuMixedTripletStep'),
)

# SEEDING LAYERS
hiRegitMuPixelLessStepSeedLayers =  RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepSeedLayers.clone(
    ComponentName = 'hiRegitMuPixelLessStepSeedLayers',
    )
hiRegitMuPixelLessStepSeedLayers.TIB.skipClusters = cms.InputTag('hiRegitMuPixelLessStepClusters')
hiRegitMuPixelLessStepSeedLayers.TID.skipClusters = cms.InputTag('hiRegitMuPixelLessStepClusters')
hiRegitMuPixelLessStepSeedLayers.TEC.skipClusters = cms.InputTag('hiRegitMuPixelLessStepClusters')


# seeding
hiRegitMuPixelLessStepSeeds     = RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepSeeds.clone()
hiRegitMuPixelLessStepSeeds.RegionFactoryPSet                                           = HiTrackingRegionFactoryFromSTAMuonsBlock.clone()
hiRegitMuPixelLessStepSeeds.ClusterCheckPSet.doClusterCheck                             = False # do not check for max number of clusters pixel or strips
hiRegitMuPixelLessStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.EscapePt        = 1.5
hiRegitMuPixelLessStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaR          = 2.0 # default = 0.2
hiRegitMuPixelLessStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaZ_Region   = 10. # this give you the length 
hiRegitMuPixelLessStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.Rescale_Dz      = 0. # max(DeltaZ_Region,Rescale_Dz*vtx->zError())
hiRegitMuPixelLessStepSeeds.OrderedHitsFactoryPSet.SeedingLayers                        = 'hiRegitMuPixelLessStepSeedLayers'


# building: feed the new-named seeds
hiRegitMuPixelLessStepTrajectoryFilter = RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepTrajectoryFilter.clone(
    ComponentName        = 'hiRegitMuPixelLessStepTrajectoryFilter'
    )
hiRegitMuPixelLessStepTrajectoryFilter.filterPset.minPt              = 1.4 # after each new hit, apply pT cut for traj w/ at least minHitsMinPt = cms.int32(3),

hiRegitMuPixelLessStepTrajectoryBuilder = RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepTrajectoryBuilder.clone(
    ComponentName        = 'hiRegitMuPixelLessStepTrajectoryBuilder',
    trajectoryFilterName = 'hiRegitMuPixelLessStepTrajectoryFilter',
    clustersToSkip = cms.InputTag('hiRegitMuPixelLessStepClusters'),
)

hiRegitMuPixelLessStepTrackCandidates        =  RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepTrackCandidates.clone(
    src               = cms.InputTag('hiRegitMuPixelLessStepSeeds'),
    TrajectoryBuilder = 'hiRegitMuPixelLessStepTrajectoryBuilder'
    )

# fitting: feed new-names
hiRegitMuPixelLessStepTracks                 = RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepTracks.clone(
    src                 = 'hiRegitMuPixelLessStepTrackCandidates'
)


hiRegitMuPixelLessStepSelector               = RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepSelector.clone( 
    src                 ='hiRegitMuPixelLessStepTracks',
    vertices            = cms.InputTag("hiSelectedVertex"),
    trackSelectors = cms.VPSet(  
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'hiRegitMuPixelLessStepLoose',
            chi2n_par = 0.5,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 5,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 3,
            d0_par1 = ( 1.5, 4.0 ),
            dz_par1 = ( 1.5, 4.0 ),
            d0_par2 = ( 1.5, 4.0 ),
            dz_par2 = ( 1.5, 4.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'hiRegitMuPixelLessStepTight',
            preFilterName = 'hiRegitMuPixelLessStepLoose',
            chi2n_par = 0.35,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 5,
            maxNumberLostLayers = 0,
            minNumber3DLayers = 3,
            d0_par1 = ( 1.2, 4.0 ),
            dz_par1 = ( 1.2, 4.0 ),
            d0_par2 = ( 1.2, 4.0 ),
            dz_par2 = ( 1.2, 4.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'hiRegitMuPixelLessStep',
            preFilterName = 'hiRegitMuPixelLessStepTight',
            chi2n_par = 0.25,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 5,
            maxNumberLostLayers = 0,
            minNumber3DLayers = 3,
            d0_par1 = ( 1., 4.0 ),
            dz_par1 = ( 1., 4.0 ),
            d0_par2 = ( 1., 4.0 ),
            dz_par2 = ( 1., 4.0 )
            ),
        ) #end of vpset

    )

hiRegitMuonPixelLessStep = cms.Sequence(hiRegitMuPixelLessStepClusters*
                                        hiRegitMuPixelLessStepSeeds*
                                        hiRegitMuPixelLessStepTrackCandidates*
                                        hiRegitMuPixelLessStepTracks*
                                        hiRegitMuPixelLessStepSelector)



