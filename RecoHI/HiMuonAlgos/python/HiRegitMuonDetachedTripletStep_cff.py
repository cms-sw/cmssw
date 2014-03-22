import FWCore.ParameterSet.Config as cms

################################################################################### 
# pp iterative tracking modified for hiOffline reco (the vertex is the one reconstructed in HI)
################################### 3rd step: low-pT and displaced tracks from pixel triplets

from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.vertexCollection = cms.InputTag("hiSelectedVertex")
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonSrc= cms.InputTag("standAloneMuons","UpdatedAtVtx")

HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.UseVertex      = True

HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.UseFixedRegion = True
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.Phi_fixed      = 0.3
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.Eta_fixed      = 0.2


###################################
from RecoHI.HiTracking.hiRegitDetachedTripletStep_cff import *

# NEW CLUSTERS (remove previously used clusters)
hiRegitMuDetachedTripletStepClusters = RecoHI.HiTracking.hiRegitDetachedTripletStep_cff.hiRegitDetachedTripletStepClusters.clone(
    oldClusterRemovalInfo = cms.InputTag("hiRegitMuPixelPairStepClusters"),
    trajectories          = cms.InputTag("hiRegitMuPixelPairStepTracks"),
    overrideTrkQuals      = cms.InputTag('hiRegitMuPixelPairStepSelector','hiRegitMuPixelPairStep'),
)

# SEEDING LAYERS
hiRegitMuDetachedTripletStepSeedLayers =  RecoHI.HiTracking.hiRegitDetachedTripletStep_cff.hiRegitDetachedTripletStepSeedLayers.clone()
hiRegitMuDetachedTripletStepSeedLayers.BPix.skipClusters = cms.InputTag('hiRegitMuDetachedTripletStepClusters')
hiRegitMuDetachedTripletStepSeedLayers.FPix.skipClusters = cms.InputTag('hiRegitMuDetachedTripletStepClusters')

# seeding
hiRegitMuDetachedTripletStepSeeds     = RecoHI.HiTracking.hiRegitDetachedTripletStep_cff.hiRegitDetachedTripletStepSeeds.clone()
hiRegitMuDetachedTripletStepSeeds.RegionFactoryPSet                                           = HiTrackingRegionFactoryFromSTAMuonsBlock.clone()
hiRegitMuDetachedTripletStepSeeds.ClusterCheckPSet.doClusterCheck                             = False # do not check for max number of clusters pixel or strips
hiRegitMuDetachedTripletStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.EscapePt        = 0.9
hiRegitMuDetachedTripletStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaR          = 2.0 # default = 0.2
hiRegitMuDetachedTripletStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaZ_Region   = 2.0 # this give you the length 
hiRegitMuDetachedTripletStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.Rescale_Dz      = 4. # max(DeltaZ_Region,Rescale_Dz*vtx->zError())
hiRegitMuDetachedTripletStepSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'hiRegitMuDetachedTripletStepSeedLayers'
from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
#hiRegitMuDetachedTripletStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.SeedComparitorPSet.ComponentName = 'LowPtClusterShapeSeedComparitor'

# building: feed the new-named seeds
hiRegitMuDetachedTripletStepTrajectoryFilter = RecoHI.HiTracking.hiRegitDetachedTripletStep_cff.hiRegitDetachedTripletStepTrajectoryFilter.clone(
    ComponentName    = 'hiRegitMuDetachedTripletStepTrajectoryFilter'
    )
hiRegitMuDetachedTripletStepTrajectoryFilter.filterPset.minPt = 0.8 # after each new hit, apply pT cut for traj w/ at least minHitsMinPt = cms.int32(3),

hiRegitMuDetachedTripletStepTrajectoryBuilder = RecoHI.HiTracking.hiRegitDetachedTripletStep_cff.hiRegitDetachedTripletStepTrajectoryBuilder.clone(
    ComponentName        = 'hiRegitMuDetachedTripletStepTrajectoryBuilder',
    trajectoryFilterName = 'hiRegitMuDetachedTripletStepTrajectoryFilter',
    clustersToSkip       = cms.InputTag('hiRegitMuDetachedTripletStepClusters')
)

hiRegitMuDetachedTripletStepTrackCandidates        =  RecoHI.HiTracking.hiRegitDetachedTripletStep_cff.hiRegitDetachedTripletStepTrackCandidates.clone(
    src               = cms.InputTag('hiRegitMuDetachedTripletStepSeeds'),
    TrajectoryBuilder = 'hiRegitMuDetachedTripletStepTrajectoryBuilder'
    )

# fitting: feed new-names
hiRegitMuDetachedTripletStepTracks                 = RecoHI.HiTracking.hiRegitDetachedTripletStep_cff.hiRegitDetachedTripletStepTracks.clone(
    src                 = 'hiRegitMuDetachedTripletStepTrackCandidates'
)


hiRegitMuDetachedTripletStepSelector               = RecoHI.HiTracking.hiRegitDetachedTripletStep_cff.hiRegitDetachedTripletStepSelector.clone( 
    src                 ='hiRegitMuDetachedTripletStepTracks',
    trackSelectors= cms.VPSet(
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiLooseMTS.clone(
            name = 'hiRegitMuDetachedTripletStepLoose',
            ),
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
            name = 'hiRegitMuDetachedTripletStepTight',
            preFilterName = 'hiRegitMuDetachedTripletStepLoose',
            ),
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
            name = 'hiRegitMuDetachedTripletStep',
            preFilterName = 'hiRegitMuDetachedTripletStepTight',
            )
        ) #end of vpset
    )


hiRegitMuonDetachedTripletStep = cms.Sequence(hiRegitMuDetachedTripletStepClusters*
                                              hiRegitMuDetachedTripletStepSeedLayers*
                                              hiRegitMuDetachedTripletStepSeeds*
                                              hiRegitMuDetachedTripletStepTrackCandidates*
                                              hiRegitMuDetachedTripletStepTracks*
                                              hiRegitMuDetachedTripletStepSelector
                                              )

