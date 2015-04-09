import FWCore.ParameterSet.Config as cms

################################################################################### 
# pp iterative tracking modified for hiOffline reco (the vertex is the one reconstructed in HI)
################################### 3rd step: low-pT and displaced tracks from pixel triplets

from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.vertexCollection = cms.InputTag("hiSelectedVertex")
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonSrc= cms.InputTag("standAloneMuons","UpdatedAtVtx")

HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.UseVertex      = True

HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.Phi_fixed     = True
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.Eta_fixed     = True
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.DeltaPhi      = 0.3
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.DeltaEta      = 0.2

###################################
from RecoTracker.IterativeTracking.DetachedTripletStep_cff import *

# NEW CLUSTERS (remove previously used clusters)
hiRegitMuDetachedTripletStepClusters = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepClusters.clone(
    trajectories          = cms.InputTag("hiRegitMuInitialStepTracks"),
    overrideTrkQuals      = cms.InputTag('hiRegitMuInitialStepSelector','hiRegitMuInitialStep'),
    TrackQuality          = cms.string('tight')
)

# SEEDING LAYERS
hiRegitMuDetachedTripletStepSeedLayers =  RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepSeedLayers.clone()
hiRegitMuDetachedTripletStepSeedLayers.BPix.skipClusters = cms.InputTag('hiRegitMuDetachedTripletStepClusters')
hiRegitMuDetachedTripletStepSeedLayers.FPix.skipClusters = cms.InputTag('hiRegitMuDetachedTripletStepClusters')

# seeding
hiRegitMuDetachedTripletStepSeeds     = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepSeeds.clone()
hiRegitMuDetachedTripletStepSeeds.RegionFactoryPSet                                           = HiTrackingRegionFactoryFromSTAMuonsBlock.clone()
hiRegitMuDetachedTripletStepSeeds.ClusterCheckPSet.doClusterCheck                             = False # do not check for max number of clusters pixel or strips
hiRegitMuDetachedTripletStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.EscapePt        = 0.9
hiRegitMuDetachedTripletStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaR          = 2.0 # default = 0.2
hiRegitMuDetachedTripletStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaZ          = 2.0 # this give you the length 
hiRegitMuDetachedTripletStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.Rescale_Dz      = 4. # max(DeltaZ_Region,Rescale_Dz*vtx->zError())
hiRegitMuDetachedTripletStepSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'hiRegitMuDetachedTripletStepSeedLayers'
from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *


# building: feed the new-named seeds
hiRegitMuDetachedTripletStepTrajectoryFilterBase = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepTrajectoryFilterBase.clone()
hiRegitMuDetachedTripletStepTrajectoryFilterBase.minPt = 0.8 # after each new hit, apply pT cut for traj w/ at least minHitsMinPt = cms.int32(3),

hiRegitMuDetachedTripletStepTrajectoryFilter = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepTrajectoryFilter.clone()
hiRegitMuDetachedTripletStepTrajectoryFilter.filters = cms.VPSet(
      cms.PSet( refToPSet_ = cms.string('hiRegitMuDetachedTripletStepTrajectoryFilterBase')),
      cms.PSet( refToPSet_ = cms.string('detachedTripletStepTrajectoryFilterShape')))

hiRegitMuDetachedTripletStepTrajectoryBuilder = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepTrajectoryBuilder.clone(
    clustersToSkip       = cms.InputTag('hiRegitMuDetachedTripletStepClusters')
)

hiRegitMuDetachedTripletStepTrackCandidates        =  RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepTrackCandidates.clone(
    src               = cms.InputTag('hiRegitMuDetachedTripletStepSeeds'),
    TrajectoryBuilder = 'hiRegitMuDetachedTripletStepTrajectoryBuilder',
    clustersToSkip = cms.InputTag("hiRegitMuDetachedTripletStepClusters")
    )

# fitting: feed new-names
hiRegitMuDetachedTripletStepTracks                 = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepTracks.clone(
    AlgorithmName = cms.string('iter6'),
    src                 = 'hiRegitMuDetachedTripletStepTrackCandidates'
)


import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
hiRegitMuDetachedTripletStepSelector               = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepSelector.clone( 
    src                 ='hiRegitMuDetachedTripletStepTracks',
    vertices            = cms.InputTag("hiSelectedVertex"),
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
           name = 'hiRegitMuDetachedTripletStepLoose',
           qualityBit = cms.string('loose'),
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'hiRegitMuDetachedTripletStepTight',
            preFilterName = 'hiRegitMuDetachedTripletStepLoose',
            qualityBit = cms.string('loose'),
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'hiRegitMuDetachedTripletStep',
            preFilterName = 'hiRegitMuDetachedTripletStepTight',
            qualityBit = cms.string('tight'),
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

