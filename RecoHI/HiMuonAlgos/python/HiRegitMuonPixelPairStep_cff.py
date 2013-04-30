import FWCore.ParameterSet.Config as cms

################################################################################### 
# pp iterative tracking modified for hiOffline reco (the vertex is the one reconstructed in HI)
################################### 2nd step: pixel pairs

from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.vertexCollection = cms.InputTag("hiSelectedVertex")
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonSrc= cms.InputTag("standAloneMuons","UpdatedAtVtx")

HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.UseVertex      = True

HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.UseFixedRegion = True
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.Phi_fixed      = 0.2
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.Eta_fixed      = 0.2


###################################
from RecoTracker.IterativeTracking.PixelPairStep_cff import *

# NEW CLUSTERS (remove previously used clusters)
hiRegitMuPixelPairStepClusters = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepClusters.clone(
    oldClusterRemovalInfo = cms.InputTag("hiRegitMuLowPtTripletStepClusters"),
    trajectories = cms.InputTag("hiRegitMuLowPtTripletStepTracks"),
    overrideTrkQuals = cms.InputTag('hiRegitMuLowPtTripletStepSelector','hiRegitMuLowPtTripletStep'),
)


# SEEDING LAYERS
hiRegitMuPixelPairStepSeedLayers =  RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepSeedLayers.clone(
    ComponentName = 'hiRegitMuPixelPairStepSeedLayers'
    )
hiRegitMuPixelPairStepSeedLayers.BPix.skipClusters = cms.InputTag('hiRegitMuPixelPairStepClusters')
hiRegitMuPixelPairStepSeedLayers.FPix.skipClusters = cms.InputTag('hiRegitMuPixelPairStepClusters')



# seeding
hiRegitMuPixelPairStepSeeds     = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepSeeds.clone()
hiRegitMuPixelPairStepSeeds.RegionFactoryPSet                                           = HiTrackingRegionFactoryFromSTAMuonsBlock.clone()
hiRegitMuPixelPairStepSeeds.ClusterCheckPSet.doClusterCheck                             = False # do not check for max number of clusters pixel or strips
hiRegitMuPixelPairStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.EscapePt        = 1.5
hiRegitMuPixelPairStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaR          = 0.01 # default = 0.2
hiRegitMuPixelPairStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaZ_Region   = 0.03 # this give you the length 
hiRegitMuPixelPairStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.Rescale_Dz      = 0. # max(DeltaZ_Region,Rescale_Dz*vtx->zError())
hiRegitMuPixelPairStepSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'hiRegitMuPixelPairStepSeedLayers'


# building: feed the new-named seeds
hiRegitMuPixelPairStepTrajectoryFilter = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepTrajectoryFilter.clone(
    ComponentName    = 'hiRegitMuPixelPairStepTrajectoryFilter'
    )
hiRegitMuPixelPairStepTrajectoryFilter.filterPset.minPt = 1.4 # after each new hit, apply pT cut for traj w/ at least minHitsMinPt = cms.int32(3),

hiRegitMuPixelPairStepTrajectoryBuilder = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepTrajectoryBuilder.clone(
    ComponentName        = 'hiRegitMuPixelPairStepTrajectoryBuilder',
    trajectoryFilterName = 'hiRegitMuPixelPairStepTrajectoryFilter',
    clustersToSkip       = cms.InputTag('hiRegitMuPixelPairStepClusters'),
)

# trackign candidate
hiRegitMuPixelPairStepTrackCandidates        =  RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepTrackCandidates.clone(
    src               = cms.InputTag('hiRegitMuPixelPairStepSeeds'),
    TrajectoryBuilder = 'hiRegitMuPixelPairStepTrajectoryBuilder'
    )

# fitting: feed new-names
hiRegitMuPixelPairStepTracks                 = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepTracks.clone(
    src                 = 'hiRegitMuPixelPairStepTrackCandidates'
)


hiRegitMuPixelPairStepSelector               = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepSelector.clone( 
    src                 ='hiRegitMuPixelPairStepTracks',
    vertices            = cms.InputTag("hiSelectedVertex"),
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'hiRegitMuPixelPairStepLoose',
            ), #end of pset
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'hiRegitMuPixelPairStepTight',
            preFilterName = 'hiRegitMuPixelPairStepLoose',
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'hiRegitMuPixelPairStep',
            preFilterName = 'hiRegitMuPixelPairStepTight',
            ),
        ) #end of vpset
)

hiRegitMuonPixelPairStep = cms.Sequence(hiRegitMuPixelPairStepClusters*
                                        hiRegitMuPixelPairStepSeeds*
                                        hiRegitMuPixelPairStepTrackCandidates*
                                        hiRegitMuPixelPairStepTracks*
                                        hiRegitMuPixelPairStepSelector)
