import FWCore.ParameterSet.Config as cms

################################################################################### 
# pp iterative tracking modified for hiOffline reco (the vertex is the one reconstructed in HI)
################################### 2nd step: pixel pairs

from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.vertexCollection = cms.InputTag("hiSelectedVertex")
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonSrc= cms.InputTag("standAloneMuons","UpdatedAtVtx")

HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.UseVertex      = True

HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.Phi_fixed     = True
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.Eta_fixed     = True
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.DeltaPhi      = 0.3
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.DeltaEta      = 0.2

###################################
from RecoTracker.IterativeTracking.PixelPairStep_cff import *

# NEW CLUSTERS (remove previously used clusters)
hiRegitMuPixelPairStepClusters = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepClusters.clone(
    trajectories          = cms.InputTag("hiRegitMuInitialStepTracks"),
		overrideTrkQuals      = cms.InputTag('hiRegitMuInitialStepSelector','hiRegitMuInitialStep'),
                trackClassifier       = cms.InputTag(''),
		oldClusterRemovalInfo = cms.InputTag(""),
		TrackQuality          = cms.string('tight')
)


# SEEDING LAYERS
hiRegitMuPixelPairStepSeedLayers =  RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepSeedLayers.clone()
hiRegitMuPixelPairStepSeedLayers.BPix.skipClusters = cms.InputTag('hiRegitMuPixelPairStepClusters')
hiRegitMuPixelPairStepSeedLayers.FPix.skipClusters = cms.InputTag('hiRegitMuPixelPairStepClusters')



# seeding
hiRegitMuPixelPairStepSeeds     = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepSeeds.clone()
hiRegitMuPixelPairStepSeeds.RegionFactoryPSet                                           = HiTrackingRegionFactoryFromSTAMuonsBlock.clone()
hiRegitMuPixelPairStepSeeds.ClusterCheckPSet.doClusterCheck                             = False # do not check for max number of clusters pixel or strips
hiRegitMuPixelPairStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.Pt_min          = 1.0
hiRegitMuPixelPairStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaR          = 0.01 # default = 0.2
hiRegitMuPixelPairStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaZ          = 0.09 # this give you the length 
hiRegitMuPixelPairStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.Rescale_Dz      = 0. # max(DeltaZ_Region,Rescale_Dz*vtx->zError())
hiRegitMuPixelPairStepSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'hiRegitMuPixelPairStepSeedLayers'


# building: feed the new-named seeds
hiRegitMuPixelPairStepTrajectoryFilterBase = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepTrajectoryFilterBase.clone()
hiRegitMuPixelPairStepTrajectoryFilterBase.minPt                = 0.8
hiRegitMuPixelPairStepTrajectoryFilterBase.minimumNumberOfHits  = 6
hiRegitMuPixelPairStepTrajectoryFilterBase.minHitsMinPt         = 4

hiRegitMuPixelPairStepTrajectoryFilter = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepTrajectoryFilter.clone()
hiRegitMuPixelPairStepTrajectoryFilter.filters = cms.VPSet(
      cms.PSet( refToPSet_ = cms.string('hiRegitMuPixelPairStepTrajectoryFilterBase')),
      cms.PSet( refToPSet_ = cms.string('pixelPairStepTrajectoryFilterShape')))


hiRegitMuPixelPairStepTrajectoryBuilder = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepTrajectoryBuilder.clone(
    trajectoryFilter = cms.PSet(
       refToPSet_ = cms.string('hiRegitMuPixelPairStepTrajectoryFilter')
       ),
    minNrOfHitsForRebuild = 6 #change from default 4
)

# trackign candidate
hiRegitMuPixelPairStepTrackCandidates        =  RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepTrackCandidates.clone(
    src               = cms.InputTag('hiRegitMuPixelPairStepSeeds'),
    TrajectoryBuilder = 'hiRegitMuPixelPairStepTrajectoryBuilder',
    clustersToSkip = cms.InputTag("hiRegitMuPixelPairStepClusters"),
    maxNSeeds         = cms.uint32(1000000)
    )

# fitting: feed new-names
hiRegitMuPixelPairStepTracks                 = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepTracks.clone(
    AlgorithmName = cms.string('hiRegitMuPixelPairStep'),
    src                 = 'hiRegitMuPixelPairStepTrackCandidates',
    clustersToSkip       = cms.InputTag('hiRegitMuPixelPairStepClusters'),
)


import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
import RecoHI.HiTracking.hiMultiTrackSelector_cfi
hiRegitMuPixelPairStepSelector = RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiMultiTrackSelector.clone(
    src                 ='hiRegitMuPixelPairStepTracks',
    vertices            = cms.InputTag("hiSelectedVertex"),
    useAnyMVA = cms.bool(True),
    GBRForestLabel = cms.string('HIMVASelectorIter6'),
    GBRForestVars = cms.vstring(['chi2perdofperlayer', 'dxyperdxyerror', 'dzperdzerror', 'nhits', 'nlayers', 'eta']),
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
           name = 'hiRegitMuPixelPairStepLoose',
           min_nhits = cms.uint32(8)
            ), #end of pset
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
            name = 'hiRegitMuPixelPairStepTight',
            preFilterName = 'hiRegitMuPixelPairStepLoose',
            min_nhits = cms.uint32(8),
            useMVA = cms.bool(True),
            minMVA = cms.double(-0.58)
            ),
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
            name = 'hiRegitMuPixelPairStep',
            preFilterName = 'hiRegitMuPixelPairStepTight',
            min_nhits = cms.uint32(8),
            useMVA = cms.bool(True),
            minMVA = cms.double(0.77)
            ),
        ) #end of vpset
)

hiRegitMuonPixelPairStep = cms.Sequence(hiRegitMuPixelPairStepClusters*
                                        hiRegitMuPixelPairStepSeedLayers*
                                        hiRegitMuPixelPairStepSeeds*
                                        hiRegitMuPixelPairStepTrackCandidates*
                                        hiRegitMuPixelPairStepTracks*
                                        hiRegitMuPixelPairStepSelector)
