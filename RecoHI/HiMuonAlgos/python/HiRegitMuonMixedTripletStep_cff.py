import FWCore.ParameterSet.Config as cms

################################################################################### 
# pp iterative tracking modified for hiOffline reco (the vertex is the one reconstructed in HI)
################################### 4th step: large impact parameter tracking using mixed-triplet seeding

from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.vertexCollection = cms.InputTag("hiSelectedVertex")
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonSrc= cms.InputTag("standAloneMuons","UpdatedAtVtx")

HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.UseVertex      = True
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.Phi_fixed     = True
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.Eta_fixed     = True
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.DeltaPhi      = 0.3
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.DeltaEta      = 0.2

###################################
from RecoTracker.IterativeTracking.MixedTripletStep_cff import *

# NEW CLUSTERS (remove previously used clusters)
hiRegitMuMixedTripletStepClusters = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepClusters.clone(
    oldClusterRemovalInfo = cms.InputTag("hiRegitMuPixelPairStepClusters"),
    trajectories          = cms.InputTag("hiRegitMuPixelPairStepTracks"),
    overrideTrkQuals      = cms.InputTag('hiRegitMuPixelPairStepSelector','hiRegitMuPixelPairStep'),
    TrackQuality          = cms.string('tight')
)


# SEEDING LAYERS A
hiRegitMuMixedTripletStepSeedLayersA =  RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSeedLayersA.clone()
hiRegitMuMixedTripletStepSeedLayersA.BPix.skipClusters = cms.InputTag('hiRegitMuMixedTripletStepClusters')
hiRegitMuMixedTripletStepSeedLayersA.FPix.skipClusters = cms.InputTag('hiRegitMuMixedTripletStepClusters')
hiRegitMuMixedTripletStepSeedLayersA.TEC.skipClusters  = cms.InputTag('hiRegitMuMixedTripletStepClusters')

# SEEDS A
hiRegitMuMixedTripletStepSeedsA = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSeedsA.clone()
hiRegitMuMixedTripletStepSeedsA.RegionFactoryPSet                                           = HiTrackingRegionFactoryFromSTAMuonsBlock.clone()
hiRegitMuMixedTripletStepSeedsA.ClusterCheckPSet.doClusterCheck                             = False # do not check for max number of clusters pixel or strips
hiRegitMuMixedTripletStepSeedsA.RegionFactoryPSet.MuonTrackingRegionBuilder.EscapePt        = 1.3
hiRegitMuMixedTripletStepSeedsA.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaR          = 0.5 # default = 0.2
hiRegitMuMixedTripletStepSeedsA.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaZ	    = 0.5 # this give you the length 
hiRegitMuMixedTripletStepSeedsA.RegionFactoryPSet.MuonTrackingRegionBuilder.Rescale_Dz      = 4.   # max(DeltaZ_Region,Rescale_Dz*vtx->zError())
hiRegitMuMixedTripletStepSeedsA.OrderedHitsFactoryPSet.SeedingLayers = 'hiRegitMuMixedTripletStepSeedLayersA'

# SEEDING LAYERS B
hiRegitMuMixedTripletStepSeedLayersB =  RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSeedLayersB.clone()
hiRegitMuMixedTripletStepSeedLayersB.BPix.skipClusters = cms.InputTag('hiRegitMuMixedTripletStepClusters')
hiRegitMuMixedTripletStepSeedLayersB.TIB.skipClusters  = cms.InputTag('hiRegitMuMixedTripletStepClusters')


hiRegitMuMixedTripletStepSeedsB = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSeedsB.clone()
hiRegitMuMixedTripletStepSeedsB.RegionFactoryPSet                                           = HiTrackingRegionFactoryFromSTAMuonsBlock.clone()
hiRegitMuMixedTripletStepSeedsB.ClusterCheckPSet.doClusterCheck                             = False # do not check for max number of clusters pixel or strips
hiRegitMuMixedTripletStepSeedsB.RegionFactoryPSet.MuonTrackingRegionBuilder.EscapePt        = 1.5
hiRegitMuMixedTripletStepSeedsB.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaR          = 0.5 # default = 0.2
hiRegitMuMixedTripletStepSeedsB.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaZ          = 0.5 # this give you the length 
hiRegitMuMixedTripletStepSeedsB.RegionFactoryPSet.MuonTrackingRegionBuilder.Rescale_Dz      = 4.   # max(DeltaZ_Region,Rescale_Dz*vtx->zError())
hiRegitMuMixedTripletStepSeedsB.OrderedHitsFactoryPSet.SeedingLayers = 'hiRegitMuMixedTripletStepSeedLayersB'

# combine seeds
hiRegitMuMixedTripletStepSeeds = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSeeds.clone(
    seedCollections = cms.VInputTag(
        cms.InputTag('hiRegitMuMixedTripletStepSeedsA'),
        cms.InputTag('hiRegitMuMixedTripletStepSeedsB'),
        )
    )

# track building
hiRegitMuMixedTripletStepTrajectoryFilter = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepTrajectoryFilter.clone()

hiRegitMuMixedTripletStepTrajectoryFilter.minPt = 1.
hiRegitMuMixedTripletStepTrajectoryFilter.minimumNumberOfHits = 6
hiRegitMuMixedTripletStepTrajectoryFilter.minHitsMinPt        = 4


 # after each new hit, apply pT cut for traj w/ at least minHitsMinPt = cms.int32(3),

hiRegitMuMixedTripletStepTrajectoryBuilder = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepTrajectoryBuilder.clone(
    trajectoryFilter = cms.PSet(
       refToPSet_ = cms.string('hiRegitMuMixedTripletStepTrajectoryFilter')
       ),
    minNrOfHitsForRebuild = 6 #change from default 4
)

hiRegitMuMixedTripletStepTrackCandidates        =  RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepTrackCandidates.clone(
    src               = cms.InputTag('hiRegitMuMixedTripletStepSeeds'),
    TrajectoryBuilderPSet = cms.PSet(
       refToPSet_ = cms.string('hiRegitMuMixedTripletStepTrajectoryBuilder')
       ),
    clustersToSkip       = cms.InputTag('hiRegitMuMixedTripletStepClusters'), 
    maxNSeeds         = cms.uint32(1000000)
    )

# fitting: feed new-names
hiRegitMuMixedTripletStepTracks                 = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepTracks.clone(
    AlgorithmName = cms.string('iter7'),
    src                 = 'hiRegitMuMixedTripletStepTrackCandidates',
)


# TRACK SELECTION AND QUALITY FLAG SETTING.
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
hiRegitMuMixedTripletStepSelector =  RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSelector.clone(
    src                 = 'hiRegitMuMixedTripletStepTracks',
    vertices            = cms.InputTag("hiSelectedVertex"),
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
           name = 'hiRegitMuMixedTripletStepLoose',
           qualityBit = cms.string('loose'),
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'hiRegitMuMixedTripletStepTight',
            preFilterName = 'hiRegitMuMixedTripletStepLoose',
            qualityBit = cms.string('loose'),
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'hiRegitMuMixedTripletStep',
            preFilterName = 'hiRegitMuMixedTripletStepTight',
            qualityBit = cms.string('tight'),
            )
        ) #end of vpset
    ) #end of clone

hiRegitMuonMixedTripletStep = cms.Sequence(hiRegitMuMixedTripletStepClusters*
                                         hiRegitMuMixedTripletStepSeedLayersA*
                                         hiRegitMuMixedTripletStepSeedsA*
                                         hiRegitMuMixedTripletStepSeedLayersB*
                                         hiRegitMuMixedTripletStepSeedsB*
                                         hiRegitMuMixedTripletStepSeeds*
                                         hiRegitMuMixedTripletStepTrackCandidates*
                                         hiRegitMuMixedTripletStepTracks*
                                         hiRegitMuMixedTripletStepSelector)
