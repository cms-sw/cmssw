import FWCore.ParameterSet.Config as cms

# pp iterative tracking modified for hiOffline reco (the vertex is the one reconstructed in HI)
################################### 1st step:pixel-triplet seeding, lower-pT;

from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.vertexCollection = cms.InputTag("hiSelectedVertex")
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonSrc= cms.InputTag("standAloneMuons","UpdatedAtVtx")

HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.UseVertex      = True
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.UseFixedRegion = True
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.Phi_fixed      = 0.3
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.Eta_fixed      = 0.2

###################################
from RecoHI.HiTracking.hiRegitLowPtTripletStep_cff import *

# remove previously used clusters
hiRegitMuLowPtTripletStepClusters = RecoHI.HiTracking.hiRegitLowPtTripletStep_cff.hiRegitLowPtTripletStepClusters.clone(
    oldClusterRemovalInfo = cms.InputTag("hiRegitMuInitialStepClusters"),
    trajectories          = cms.InputTag("hiRegitMuInitialStepTracks"),
    overrideTrkQuals      = cms.InputTag('hiRegitMuInitialStepSelector','hiRegitMuInitialStep'),
)

# SEEDING LAYERS
hiRegitMuLowPtTripletStepSeedLayers = RecoHI.HiTracking.hiRegitLowPtTripletStep_cff.hiRegitLowPtTripletStepSeedLayers.clone()
hiRegitMuLowPtTripletStepSeedLayers.BPix.skipClusters = cms.InputTag('hiRegitMuLowPtTripletStepClusters')
hiRegitMuLowPtTripletStepSeedLayers.FPix.skipClusters = cms.InputTag('hiRegitMuLowPtTripletStepClusters')

# seeds
hiRegitMuLowPtTripletStepSeeds     = RecoHI.HiTracking.hiRegitLowPtTripletStep_cff.hiRegitLowPtTripletStepSeeds.clone()
hiRegitMuLowPtTripletStepSeeds.RegionFactoryPSet                                           = HiTrackingRegionFactoryFromSTAMuonsBlock.clone()
hiRegitMuLowPtTripletStepSeeds.ClusterCheckPSet.doClusterCheck                             = False # do not check for max number of clusters pixel or strips
hiRegitMuLowPtTripletStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.EscapePt        = 0.9
hiRegitMuLowPtTripletStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaR          = 1. # default = 0.2
hiRegitMuLowPtTripletStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaZ_Region   = 1. # this give you the length 
hiRegitMuLowPtTripletStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.Rescale_Dz      = 4. # max(DeltaZ_Region,Rescale_Dz*vtx->zError())
hiRegitMuLowPtTripletStepSeeds.OrderedHitsFactoryPSet.SeedingLayers                                  = 'hiRegitMuLowPtTripletStepSeedLayers'
from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
hiRegitMuLowPtTripletStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.SeedComparitorPSet.ComponentName = 'LowPtClusterShapeSeedComparitor'


# building: feed the new-named seeds
hiRegitMuLowPtTripletStepTrajectoryFilter = RecoHI.HiTracking.hiRegitLowPtTripletStep_cff.hiRegitLowPtTripletStepTrajectoryFilter.clone(
    ComponentName = 'hiRegitMuLowPtTripletStepTrajectoryFilter'
    )
hiRegitMuLowPtTripletStepTrajectoryFilter.filterPset.minPt              = 0.8 # after each new hit, apply pT cut for traj w/ at least minHitsMinPt = cms.int32(3),


hiRegitMuLowPtTripletStepTrajectoryBuilder = RecoHI.HiTracking.hiRegitLowPtTripletStep_cff.hiRegitLowPtTripletStepTrajectoryBuilder.clone(
    ComponentName        = 'hiRegitMuLowPtTripletStepTrajectoryBuilder',
    trajectoryFilterName = 'hiRegitMuLowPtTripletStepTrajectoryFilter',
    clustersToSkip = cms.InputTag('hiRegitMuLowPtTripletStepClusters'),
)

# track candidates
hiRegitMuLowPtTripletStepTrackCandidates        =  RecoHI.HiTracking.hiRegitLowPtTripletStep_cff.hiRegitLowPtTripletStepTrackCandidates.clone(
    src               = cms.InputTag('hiRegitMuLowPtTripletStepSeeds'),
    TrajectoryBuilder = 'hiRegitMuLowPtTripletStepTrajectoryBuilder',
    maxNSeeds         = cms.uint32(1000000)
    )

# fitting: feed new-names
hiRegitMuLowPtTripletStepTracks                 = RecoHI.HiTracking.hiRegitLowPtTripletStep_cff.hiRegitLowPtTripletStepTracks.clone(
    src                 = 'hiRegitMuLowPtTripletStepTrackCandidates'
)


hiRegitMuLowPtTripletStepSelector               = RecoHI.HiTracking.hiRegitLowPtTripletStep_cff.hiRegitLowPtTripletStepSelector.clone( 
    src                 ='hiRegitMuLowPtTripletStepTracks',
    vertices            = cms.InputTag("hiSelectedVertex"),
    trackSelectors= cms.VPSet(
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiLooseMTS.clone(
            name = 'hiRegitMuLowPtTripletStepLoose',
         #   minNumberLayers = 10
            ), #end of pset
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
            name = 'hiRegitMuLowPtTripletStepTight',
            preFilterName = 'hiRegitMuLowPtTripletStepLoose',
          #  minNumberLayers = 10
            ),
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
            name = 'hiRegitMuLowPtTripletStep',
            preFilterName = 'hiRegitMuLowPtTripletStepTight',
           # minNumberLayers = 10
            ),
        ) #end of vpset
)

hiRegitMuonLowPtTripletStep = cms.Sequence(hiRegitMuLowPtTripletStepClusters*
                                           hiRegitMuLowPtTripletStepSeedLayers*
                                           hiRegitMuLowPtTripletStepSeeds*
                                           hiRegitMuLowPtTripletStepTrackCandidates*
                                           hiRegitMuLowPtTripletStepTracks*
                                           hiRegitMuLowPtTripletStepSelector)



