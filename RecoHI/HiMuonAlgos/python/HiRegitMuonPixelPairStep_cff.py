import FWCore.ParameterSet.Config as cms

################################################################################### 
# pp iterative tracking modified for hiOffline reco (the vertex is the one reconstructed in HI)
################################### 2nd step: pixel pairs

from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.vertexCollection = cms.InputTag("hiSelectedVertex")
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonSrc= cms.InputTag("standAloneMuons","UpdatedAtVtx")

HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.UseVertex      = True

HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.UseFixedRegion = True
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.Phi_fixed      = 0.3
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.Eta_fixed      = 0.2


###################################
from RecoHI.HiTracking.hiRegitPixelPairStep_cff import *

# NEW CLUSTERS (remove previously used clusters)
hiRegitMuPixelPairStepClusters = RecoHI.HiTracking.hiRegitPixelPairStep_cff.hiRegitPixelPairStepClusters.clone(
    oldClusterRemovalInfo = cms.InputTag("hiRegitMuLowPtTripletStepClusters"),
    trajectories = cms.InputTag("hiRegitMuLowPtTripletStepTracks"),
    overrideTrkQuals = cms.InputTag('hiRegitMuLowPtTripletStepSelector','hiRegitMuLowPtTripletStep'),
)


# SEEDING LAYERS
hiRegitMuPixelPairStepSeedLayers =   RecoHI.HiTracking.hiRegitPixelPairStep_cff.hiRegitPixelPairStepSeedLayers.clone()
hiRegitMuPixelPairStepSeedLayers.BPix.skipClusters = cms.InputTag('hiRegitMuPixelPairStepClusters')
hiRegitMuPixelPairStepSeedLayers.FPix.skipClusters = cms.InputTag('hiRegitMuPixelPairStepClusters')



# seeding
hiRegitMuPixelPairStepSeeds     =  RecoHI.HiTracking.hiRegitPixelPairStep_cff.hiRegitPixelPairStepSeeds.clone()
hiRegitMuPixelPairStepSeeds.RegionFactoryPSet                                           = HiTrackingRegionFactoryFromSTAMuonsBlock.clone()
hiRegitMuPixelPairStepSeeds.ClusterCheckPSet.doClusterCheck                             = False # do not check for max number of clusters pixel or strips
hiRegitMuPixelPairStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.EscapePt        = 1.0
hiRegitMuPixelPairStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaR          = 0.01 # default = 0.2
hiRegitMuPixelPairStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaZ_Region   = 0.09 # this give you the length 
hiRegitMuPixelPairStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.Rescale_Dz      = 0. # max(DeltaZ_Region,Rescale_Dz*vtx->zError())
hiRegitMuPixelPairStepSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'hiRegitMuPixelPairStepSeedLayers'


# building: feed the new-named seeds
hiRegitMuPixelPairStepTrajectoryFilter =  RecoHI.HiTracking.hiRegitPixelPairStep_cff.hiRegitPixelPairStepTrajectoryFilter.clone()
hiRegitMuPixelPairStepTrajectoryFilter.minPt                = 0.8
hiRegitMuPixelPairStepTrajectoryFilter.minimumNumberOfHits  = 6
hiRegitMuPixelPairStepTrajectoryFilter.minHitsMinPt         = 4



hiRegitMuPixelPairStepTrajectoryBuilder =  RecoHI.HiTracking.hiRegitPixelPairStep_cff.hiRegitPixelPairStepTrajectoryBuilder.clone(
    trajectoryFilter     = cms.PSet(refToPSet_ = cms.string('hiRegitMuPixelPairStepTrajectoryFilter')),
    clustersToSkip       = cms.InputTag('hiRegitMuPixelPairStepClusters'),
    minNrOfHitsForRebuild = 6 #change from default 4
)

# trackign candidate
hiRegitMuPixelPairStepTrackCandidates        = RecoHI.HiTracking.hiRegitPixelPairStep_cff.hiRegitPixelPairStepTrackCandidates.clone(
    src               = cms.InputTag('hiRegitMuPixelPairStepSeeds'),
    TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string('hiRegitMuPixelPairStepTrajectoryBuilder')),
    maxNSeeds         = cms.uint32(1000000)
    )

# fitting: feed new-names
hiRegitMuPixelPairStepTracks                 =  RecoHI.HiTracking.hiRegitPixelPairStep_cff.hiRegitPixelPairStepTracks.clone(
    src                 = 'hiRegitMuPixelPairStepTrackCandidates'
)


hiRegitMuPixelPairStepSelector               =  RecoHI.HiTracking.hiRegitPixelPairStep_cff.hiRegitPixelPairStepSelector.clone( 
    src                 ='hiRegitMuPixelPairStepTracks',
    trackSelectors= cms.VPSet(
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiLooseMTS.clone(
            name = 'hiRegitMuPixelPairStepLoose',
             ), #end of pset
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
            name = 'hiRegitMuPixelPairStepTight',
            preFilterName = 'hiRegitMuPixelPairStepLoose',
            ),
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
            name = 'hiRegitMuPixelPairStep',
            preFilterName = 'hiRegitMuPixelPairStepTight',
            #      minNumberLayers = 10
            ),
        ) #end of vpset
    )

hiRegitMuonPixelPairStep = cms.Sequence(hiRegitMuPixelPairStepClusters*
                                        hiRegitMuPixelPairStepSeedLayers*
                                        hiRegitMuPixelPairStepSeeds*
                                        hiRegitMuPixelPairStepTrackCandidates*
                                        hiRegitMuPixelPairStepTracks*
                                        hiRegitMuPixelPairStepSelector)
