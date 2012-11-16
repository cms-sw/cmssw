import FWCore.ParameterSet.Config as cms

# pp iterative tracking modified for hiOffline reco (the vertex is the one reconstructed in HI)
################################### 1st step:pixel-triplet seeding, lower-pT;

from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.vertexCollection = cms.InputTag("hiSelectedVertex")
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonSrc= cms.InputTag("standAloneMuons","UpdatedAtVtx")

HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.UseVertex      = True

HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.UseFixedRegion = True
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.Phi_fixed      = 0.2
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.Eta_fixed      = 0.2

###################################
from RecoTracker.IterativeTracking.LowPtTripletStep_cff import *

# remove previously used clusters
hiRegitMuLowPtTripletStepClusters = RecoTracker.IterativeTracking.LowPtTripletStep_cff.lowPtTripletStepClusters.clone(
    trajectories     = cms.InputTag("hiRegitMuInitialStepTracks"),
    overrideTrkQuals = cms.InputTag('hiRegitMuInitialStepSelector','hiRegitMuInitialStep'),
)

# SEEDING LAYERS
hiRegitMuLowPtTripletStepSeedLayers =  RecoTracker.IterativeTracking.LowPtTripletStep_cff.lowPtTripletStepSeedLayers.clone(
    ComponentName = 'hiRegitMuLowPtTripletStepSeedLayers'
    )
hiRegitMuLowPtTripletStepSeedLayers.BPix.skipClusters = cms.InputTag('hiRegitMuLowPtTripletStepClusters')
hiRegitMuLowPtTripletStepSeedLayers.FPix.skipClusters = cms.InputTag('hiRegitMuLowPtTripletStepClusters')

# seeds
hiRegitMuLowPtTripletStepSeeds     = RecoTracker.IterativeTracking.LowPtTripletStep_cff.lowPtTripletStepSeeds.clone()
hiRegitMuLowPtTripletStepSeeds.RegionFactoryPSet                                           = HiTrackingRegionFactoryFromSTAMuonsBlock.clone()
hiRegitMuLowPtTripletStepSeeds.ClusterCheckPSet.doClusterCheck                             = False # do not check for max number of clusters pixel or strips
hiRegitMuLowPtTripletStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.EscapePt        = 1.
hiRegitMuLowPtTripletStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaR          = 0.03 # default = 0.2
hiRegitMuLowPtTripletStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaZ_Region   = 0. # this give you the length 
hiRegitMuLowPtTripletStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.Rescale_Dz      = 4. # max(DeltaZ_Region,Rescale_Dz*vtx->zError())
hiRegitMuLowPtTripletStepSeeds.OrderedHitsFactoryPSet.SeedingLayers                                  = 'hiRegitMuLowPtTripletStepSeedLayers'
from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
hiRegitMuLowPtTripletStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.SeedComparitorPSet.ComponentName = 'LowPtClusterShapeSeedComparitor'


# building: feed the new-named seeds
hiRegitMuLowPtTripletStepTrajectoryFilter = RecoTracker.IterativeTracking.LowPtTripletStep_cff.lowPtTripletStepTrajectoryFilter.clone(
    ComponentName = 'hiRegitMuLowPtTripletStepTrajectoryFilter'
    )
hiRegitMuLowPtTripletStepTrajectoryFilter.filterPset.minPt              = 1.4 # after each new hit, apply pT cut for traj w/ at least minHitsMinPt = cms.int32(3),


hiRegitMuLowPtTripletStepTrajectoryBuilder = RecoTracker.IterativeTracking.LowPtTripletStep_cff.lowPtTripletStepTrajectoryBuilder.clone(
    ComponentName        = 'hiRegitMuLowPtTripletStepTrajectoryBuilder',
    trajectoryFilterName = 'hiRegitMuLowPtTripletStepTrajectoryFilter',
    clustersToSkip = cms.InputTag('hiRegitMuLowPtTripletStepClusters'),
)

# track candidates
hiRegitMuLowPtTripletStepTrackCandidates        =  RecoTracker.IterativeTracking.LowPtTripletStep_cff.lowPtTripletStepTrackCandidates.clone(
    src               = cms.InputTag('hiRegitMuLowPtTripletStepSeeds'),
    TrajectoryBuilder = 'hiRegitMuLowPtTripletStepTrajectoryBuilder'
    )

# fitting: feed new-names
hiRegitMuLowPtTripletStepTracks                 = RecoTracker.IterativeTracking.LowPtTripletStep_cff.lowPtTripletStepTracks.clone(
    src                 = 'hiRegitMuLowPtTripletStepTrackCandidates'
)


hiRegitMuLowPtTripletStepSelector               = RecoTracker.IterativeTracking.LowPtTripletStep_cff.lowPtTripletStepSelector.clone( 
    src                 ='hiRegitMuLowPtTripletStepTracks',
    vertices            = cms.InputTag("hiSelectedVertex"),
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'hiRegitMuLowPtTripletStepLoose',
            ), #end of pset
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'hiRegitMuLowPtTripletStepTight',
            preFilterName = 'hiRegitMuLowPtTripletStepLoose',
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'hiRegitMuLowPtTripletStep',
            preFilterName = 'hiRegitMuLowPtTripletStepTight',
            ),
        ) #end of vpset
)

hiRegitMuonLowPtTripletStep = cms.Sequence(hiRegitMuLowPtTripletStepClusters*
                                           hiRegitMuLowPtTripletStepSeeds*
                                           hiRegitMuLowPtTripletStepTrackCandidates*
                                           hiRegitMuLowPtTripletStepTracks*
                                           hiRegitMuLowPtTripletStepSelector)



