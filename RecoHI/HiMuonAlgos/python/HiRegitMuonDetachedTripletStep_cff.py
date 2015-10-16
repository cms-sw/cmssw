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
    trajectories          = cms.InputTag("hiRegitMuPixelLessStepTracks"),
    overrideTrkQuals      = cms.InputTag('hiRegitMuPixelLessStepSelector','hiRegitMuPixelLessStep'),
    trackClassifier       = cms.InputTag(''),
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
hiRegitMuDetachedTripletStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.Pt_min          = 0.9
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
    AlgorithmName = cms.string('hiRegitMuDetachedTripletStep'),
    src                 = 'hiRegitMuDetachedTripletStepTrackCandidates'
)


import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
import RecoHI.HiTracking.hiMultiTrackSelector_cfi
hiRegitMuDetachedTripletStepSelector = RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiMultiTrackSelector.clone(
    src                 ='hiRegitMuDetachedTripletStepTracks',
    vertices            = cms.InputTag("hiSelectedVertex"),
    useAnyMVA = cms.bool(True),
    GBRForestLabel = cms.string('HIMVASelectorIter7'),
    GBRForestVars = cms.vstring(['chi2perdofperlayer', 'nhits', 'nlayers', 'eta']),
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
           name = 'hiRegitMuDetachedTripletStepLoose',
           min_nhits = cms.uint32(8)
            ),
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
            name = 'hiRegitMuDetachedTripletStepTight',
            preFilterName = 'hiRegitMuDetachedTripletStepLoose',
            min_nhits = cms.uint32(8),
            useMVA = cms.bool(True),
            minMVA = cms.double(-0.2)
            ),
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
            name = 'hiRegitMuDetachedTripletStep',
            preFilterName = 'hiRegitMuDetachedTripletStepTight',
            min_nhits = cms.uint32(8),
            useMVA = cms.bool(True),
            minMVA = cms.double(-0.09)
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

