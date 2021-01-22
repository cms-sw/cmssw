import FWCore.ParameterSet.Config as cms

################################################################################### 
# pp iterative tracking modified for hiOffline reco (the vertex is the one reconstructed in HI)
################################### 2nd step: pixel pairs

from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *
# Are the following values set to the same in every iteration? If yes,
# why not making the change in HITrackingRegionProducer_cfi directly
# once for all?
hiRegitMuPixelPairStepTrackingRegions = HiTrackingRegionFactoryFromSTAMuonsEDProducer.clone(
    MuonSrc = "standAloneMuons:UpdatedAtVtx", # this is the same as default, why repeat?
    MuonTrackingRegionBuilder = dict(
        vertexCollection = "hiSelectedPixelVertex",
        UseVertex     = True,
        Phi_fixed     = True,
        Eta_fixed     = True,
        DeltaPhi      = 0.3,
        DeltaEta      = 0.2,
        # Ok, the following ones are specific to PixelPairStep
        Pt_min          = 1.0,
        DeltaR          = 0.01, # default = 0.2
        DeltaZ          = 0.09, # this give you the length
        Rescale_Dz      = 0.    # max(DeltaZ_Region,Rescale_Dz*vtx->zError())
    )
)

###################################
from RecoTracker.IterativeTracking.PixelPairStep_cff import *

# NEW CLUSTERS (remove previously used clusters)
hiRegitMuPixelPairStepClusters = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepClusters.clone(
    trajectories          = "hiRegitMuInitialStepTracks",
    overrideTrkQuals      = 'hiRegitMuInitialStepSelector:hiRegitMuInitialStep',
    trackClassifier       = '',
    oldClusterRemovalInfo = "",
    TrackQuality          = 'tight'
)


# SEEDING LAYERS
hiRegitMuPixelPairStepSeedLayers =  RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepSeedLayers.clone(
    BPix = dict(skipClusters = 'hiRegitMuPixelPairStepClusters'),
    FPix = dict(skipClusters = 'hiRegitMuPixelPairStepClusters')
)

# seeding
hiRegitMuPixelPairStepHitDoublets = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepHitDoublets.clone(
    seedingLayers   = "hiRegitMuPixelPairStepSeedLayers",
    trackingRegions = "hiRegitMuPixelPairStepTrackingRegions",
    clusterCheck    = "hiRegitMuClusterCheck",
)

hiRegitMuPixelPairStepSeeds     = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepSeedsA.clone(
    seedingHitSets = "hiRegitMuPixelPairStepHitDoublets"
)


# building: feed the new-named seeds
hiRegitMuPixelPairStepTrajectoryFilterBase = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepTrajectoryFilterBase.clone(
    minPt                = 0.8,
    minimumNumberOfHits  = 6,
    minHitsMinPt         = 4
)
hiRegitMuPixelPairStepTrajectoryFilter = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepTrajectoryFilter.clone(
    filters = cms.VPSet(
      cms.PSet( refToPSet_ = cms.string('hiRegitMuPixelPairStepTrajectoryFilterBase')),
      cms.PSet( refToPSet_ = cms.string('pixelPairStepTrajectoryFilterShape')))
)

hiRegitMuPixelPairStepTrajectoryBuilder = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepTrajectoryBuilder.clone(
    trajectoryFilter = cms.PSet(
       refToPSet_ = cms.string('hiRegitMuPixelPairStepTrajectoryFilter')
       ),
    minNrOfHitsForRebuild = 6 #change from default 4
)

# trackign candidate
hiRegitMuPixelPairStepTrackCandidates = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepTrackCandidates.clone(
    src               = 'hiRegitMuPixelPairStepSeeds',
    TrajectoryBuilder = 'hiRegitMuPixelPairStepTrajectoryBuilder',
    clustersToSkip    = "hiRegitMuPixelPairStepClusters",
    maxNSeeds         = 1000000
)

# fitting: feed new-names
hiRegitMuPixelPairStepTracks = RecoTracker.IterativeTracking.PixelPairStep_cff.pixelPairStepTracks.clone(
    AlgorithmName   = 'hiRegitMuPixelPairStep',
    src             = 'hiRegitMuPixelPairStepTrackCandidates',
    clustersToSkip  = cms.InputTag('hiRegitMuPixelPairStepClusters'),
)


import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
import RecoHI.HiTracking.hiMultiTrackSelector_cfi
hiRegitMuPixelPairStepSelector = RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiMultiTrackSelector.clone(
    src            ='hiRegitMuPixelPairStepTracks',
    vertices       = "hiSelectedPixelVertex",
    useAnyMVA      = True,
    GBRForestLabel = 'HIMVASelectorIter6',
    GBRForestVars  = ['chi2perdofperlayer', 'dxyperdxyerror', 'dzperdzerror', 'nhits', 'nlayers', 'eta'],
    trackSelectors = cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name      = 'hiRegitMuPixelPairStepLoose',
            min_nhits = 8
            ), 
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
            name          = 'hiRegitMuPixelPairStepTight',
            preFilterName = 'hiRegitMuPixelPairStepLoose',
            min_nhits     = 8,
            useMVA        = True,
            minMVA        = -0.58
            ),
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
            name          = 'hiRegitMuPixelPairStep',
            preFilterName = 'hiRegitMuPixelPairStepTight',
            min_nhits     = 8,
            useMVA        = True,
            minMVA        = 0.77
            ),
        ) #end of vpset
)
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
trackingPhase1.toModify(hiRegitMuPixelPairStepSelector, useAnyMVA = False)
trackingPhase1.toModify(hiRegitMuPixelPairStepSelector, trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
           name      = 'hiRegitMuPixelPairStepLoose',
           min_nhits = 8
            ), 
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
            name          = 'hiRegitMuPixelPairStepTight',
            preFilterName = 'hiRegitMuPixelPairStepLoose',
            min_nhits     = 8,
            useMVA        = False,
            minMVA        = -0.58
            ),
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
            name          = 'hiRegitMuPixelPairStep',
            preFilterName = 'hiRegitMuPixelPairStepTight',
            min_nhits     = 8,
            useMVA        = False,
            minMVA        = 0.77
            ),
        ) #end of vpset
)

hiRegitMuonPixelPairStepTask = cms.Task(hiRegitMuPixelPairStepClusters,
                                        hiRegitMuPixelPairStepSeedLayers,
                                        hiRegitMuPixelPairStepTrackingRegions,
                                        hiRegitMuPixelPairStepHitDoublets,
                                        hiRegitMuPixelPairStepSeeds,
                                        hiRegitMuPixelPairStepTrackCandidates,
                                        hiRegitMuPixelPairStepTracks,
                                        hiRegitMuPixelPairStepSelector)
hiRegitMuonPixelPairStep = cms.Sequence(hiRegitMuonPixelPairStepTask)
