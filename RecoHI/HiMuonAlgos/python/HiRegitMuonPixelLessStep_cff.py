import FWCore.ParameterSet.Config as cms

# pp iterative tracking modified for hiOffline reco (the vertex is the one reconstructed in HI)
################################### 5th step: large impact parameter tracking using TIB/TID/TEC stereo layer seeding

from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.vertexCollection = cms.InputTag("hiSelectedVertex")
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonSrc= cms.InputTag("standAloneMuons","UpdatedAtVtx")

HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.UseVertex      = True

HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.UseFixedRegion = True
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.Phi_fixed      = 0.2
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.Eta_fixed      = 0.1

###################################
from RecoTracker.IterativeTracking.PixelLessStep_cff import *

# remove previously used clusters
hiRegitMuPixelLessStepClusters = RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepClusters.clone(
    oldClusterRemovalInfo = cms.InputTag("hiRegitMuMixedTripletStepClusters"),
    trajectories     = cms.InputTag("hiRegitMuMixedTripletStepTracks"),
    overrideTrkQuals = cms.InputTag('hiRegitMuMixedTripletStepSelector','hiRegitMuMixedTripletStep'),
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
hiRegitMuPixelLessStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.EscapePt        = 2.0
hiRegitMuPixelLessStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaR          = 0.2 # default = 0.2
hiRegitMuPixelLessStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaZ_Region   = 0.2 # this give you the length 
hiRegitMuPixelLessStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.Rescale_Dz      = 4. # max(DeltaZ_Region,Rescale_Dz*vtx->zError())
hiRegitMuPixelLessStepSeeds.OrderedHitsFactoryPSet.SeedingLayers                        = 'hiRegitMuPixelLessStepSeedLayers'


# building: feed the new-named seeds
hiRegitMuPixelLessStepTrajectoryFilter = RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepTrajectoryFilter.clone(
    ComponentName        = 'hiRegitMuPixelLessStepTrajectoryFilter',
    )
hiRegitMuPixelLessStepTrajectoryFilter.filterPset.minPt                = 1.7
hiRegitMuPixelLessStepTrajectoryFilter.filterPset.minimumNumberOfHits  = 6
hiRegitMuPixelLessStepTrajectoryFilter.filterPset.minHitsMinPt         = 4

hiRegitMuPixelLessStepTrajectoryBuilder = RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepTrajectoryBuilder.clone(
    ComponentName        = 'hiRegitMuPixelLessStepTrajectoryBuilder',
    trajectoryFilterName = 'hiRegitMuPixelLessStepTrajectoryFilter',
    clustersToSkip       = cms.InputTag('hiRegitMuPixelLessStepClusters'),
    minNrOfHitsForRebuild = 6 #change from default 4
)

hiRegitMuPixelLessStepTrackCandidates        =  RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepTrackCandidates.clone(
    src               = cms.InputTag('hiRegitMuPixelLessStepSeeds'),
    TrajectoryBuilder = 'hiRegitMuPixelLessStepTrajectoryBuilder',
    maxNSeeds         = cms.uint32(1000000)
    )

# fitting: feed new-names
hiRegitMuPixelLessStepTracks                 = RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepTracks.clone(
    src                 = 'hiRegitMuPixelLessStepTrackCandidates'
)

import RecoHI.HiTracking.hiMultiTrackSelector_cfi
hiRegitMuPixelLessStepSelector               = RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepSelector.clone( 
    src                 ='hiRegitMuPixelLessStepTracks',
    vertices            = cms.InputTag("hiSelectedVertex"),
    trackSelectors = cms.VPSet(  
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiLooseMTS.clone(
            name = 'hiRegitMuPixelLessStepLoose',
            ),
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
            name = 'hiRegitMuPixelLessStepTight',
            preFilterName = 'hiRegitMuPixelLessStepLoose',
            ),
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
            name = 'hiRegitMuPixelLessStep',
            preFilterName = 'hiRegitMuPixelLessStepTight',
            ),
        ) #end of vpset
)

hiRegitMuonPixelLessStep = cms.Sequence(hiRegitMuPixelLessStepClusters*
                                        hiRegitMuPixelLessStepSeeds*
                                        hiRegitMuPixelLessStepTrackCandidates*
                                        hiRegitMuPixelLessStepTracks*
                                        hiRegitMuPixelLessStepSelector)



