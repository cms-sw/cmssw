import FWCore.ParameterSet.Config as cms

# pp iterative tracking modified for hiOffline reco (the vertex is the one reconstructed in HI)
################################### 5th step: large impact parameter tracking using TIB/TID/TEC stereo layer seeding

from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.vertexCollection = cms.InputTag("hiSelectedVertex")
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonSrc= cms.InputTag("standAloneMuons","UpdatedAtVtx")

HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.UseVertex      = True

HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.Phi_fixed     = True
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.Eta_fixed     = True
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.DeltaPhi      = 0.2
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.DeltaEta      = 0.1

###################################
from RecoTracker.IterativeTracking.PixelLessStep_cff import *

# remove previously used clusters
hiRegitMuPixelLessStepClusters = RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepClusters.clone(
    oldClusterRemovalInfo = cms.InputTag("hiRegitMuMixedTripletStepClusters"),
    trajectories     = cms.InputTag("hiRegitMuMixedTripletStepTracks"),
    overrideTrkQuals = cms.InputTag('hiRegitMuMixedTripletStepSelector','hiRegitMuMixedTripletStep'),
    TrackQuality          = cms.string('tight')
)

# SEEDING LAYERS
hiRegitMuPixelLessStepSeedLayers =  RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepSeedLayers.clone()
hiRegitMuPixelLessStepSeedLayers.TIB.skipClusters = cms.InputTag('hiRegitMuPixelLessStepClusters')
hiRegitMuPixelLessStepSeedLayers.TID.skipClusters = cms.InputTag('hiRegitMuPixelLessStepClusters')
hiRegitMuPixelLessStepSeedLayers.TEC.skipClusters = cms.InputTag('hiRegitMuPixelLessStepClusters')
hiRegitMuPixelLessStepSeedLayers.MTIB.skipClusters = cms.InputTag('hiRegitMuPixelLessStepClusters')
hiRegitMuPixelLessStepSeedLayers.MTID.skipClusters = cms.InputTag('hiRegitMuPixelLessStepClusters')
hiRegitMuPixelLessStepSeedLayers.MTEC.skipClusters = cms.InputTag('hiRegitMuPixelLessStepClusters')


# seeding
hiRegitMuPixelLessStepSeeds     = RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepSeeds.clone()
hiRegitMuPixelLessStepSeeds.RegionFactoryPSet                                           = HiTrackingRegionFactoryFromSTAMuonsBlock.clone()
hiRegitMuPixelLessStepSeeds.ClusterCheckPSet.doClusterCheck                             = False # do not check for max number of clusters pixel or strips
hiRegitMuPixelLessStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.Pt_min          = 2.0
hiRegitMuPixelLessStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaR          = 0.2 # default = 0.2
hiRegitMuPixelLessStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaZ          = 0.2 # this give you the length 
hiRegitMuPixelLessStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.Rescale_Dz      = 4. # max(DeltaZ_Region,Rescale_Dz*vtx->zError())
hiRegitMuPixelLessStepSeeds.OrderedHitsFactoryPSet.SeedingLayers                        = 'hiRegitMuPixelLessStepSeedLayers'


# building: feed the new-named seeds
hiRegitMuPixelLessStepTrajectoryFilter = RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepTrajectoryFilter.clone()
hiRegitMuPixelLessStepTrajectoryFilter.minPt                = 1.7
hiRegitMuPixelLessStepTrajectoryFilter.minimumNumberOfHits  = 6
hiRegitMuPixelLessStepTrajectoryFilter.minHitsMinPt         = 4

hiRegitMuPixelLessStepTrajectoryBuilder = RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepTrajectoryBuilder.clone(
    trajectoryFilter = cms.PSet(
       refToPSet_ = cms.string('hiRegitMuPixelLessStepTrajectoryFilter')
       ),
    minNrOfHitsForRebuild = 6 #change from default 4
)

hiRegitMuPixelLessStepTrackCandidates        =  RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepTrackCandidates.clone(
    src               = cms.InputTag('hiRegitMuPixelLessStepSeeds'),
    TrajectoryBuilderPSet = cms.PSet(
       refToPSet_ = cms.string('hiRegitMuPixelLessStepTrajectoryBuilder')
       ),
    clustersToSkip       = cms.InputTag('hiRegitMuPixelLessStepClusters'),
    maxNSeeds         = cms.uint32(1000000)
    )

# fitting: feed new-names
hiRegitMuPixelLessStepTracks                 = RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepTracks.clone(
    AlgorithmName = cms.string('hiRegitMuPixelLessStep'),
    src                 = 'hiRegitMuPixelLessStepTrackCandidates'
)

import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
import RecoHI.HiTracking.hiMultiTrackSelector_cfi
hiRegitMuPixelLessStepSelector = RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiMultiTrackSelector.clone(
    src                 ='hiRegitMuPixelLessStepTracks',
    vertices            = cms.InputTag("hiSelectedVertex"),
    useAnyMVA = cms.bool(True),
    GBRForestLabel = cms.string('HIMVASelectorIter7'),
    GBRForestVars = cms.vstring(['chi2perdofperlayer', 'nhits', 'nlayers', 'eta']),
    trackSelectors = cms.VPSet(  
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
           name = 'hiRegitMuPixelLessStepLoose',
           min_nhits = cms.uint32(8)
            ),
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
            name = 'hiRegitMuPixelLessStepTight',
            preFilterName = 'hiRegitMuPixelLessStepLoose',
            min_nhits = cms.uint32(8),
            useMVA = cms.bool(True),
            minMVA = cms.double(-0.2)
            ),
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
            name = 'hiRegitMuPixelLessStep',
            preFilterName = 'hiRegitMuPixelLessStepTight',
            min_nhits = cms.uint32(8),
            useMVA = cms.bool(True),
            minMVA = cms.double(-0.09)
            ),
        ) #end of vpset
)

hiRegitMuonPixelLessStep = cms.Sequence(hiRegitMuPixelLessStepClusters*
                                        hiRegitMuPixelLessStepSeedLayers*
                                        hiRegitMuPixelLessStepSeeds*
                                        hiRegitMuPixelLessStepTrackCandidates*
                                        hiRegitMuPixelLessStepTracks*
                                        hiRegitMuPixelLessStepSelector)



