import FWCore.ParameterSet.Config as cms

################################################################################### 
# pp iterative tracking modified for hiOffline reco (the vertex is the one reconstructed in HI)
################################### 3rd step: low-pT and displaced tracks from pixel triplets

from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *
# Are the following values set to the same in every iteration? If yes,
# why not making the change in HITrackingRegionProducer_cfi directly
# once for all?
hiRegitMuDetachedTripletStepTrackingRegions = HiTrackingRegionFactoryFromSTAMuonsEDProducer.clone(
    MuonSrc = "standAloneMuons:UpdatedAtVtx", # this is the same as default, why repeat?
    MuonTrackingRegionBuilder = dict(
        vertexCollection = "hiSelectedPixelVertex",
        UseVertex     = True,
        Phi_fixed     = True,
        Eta_fixed     = True,
        DeltaPhi      = 0.3,
        DeltaEta      = 0.2,
        # Ok, the following ones are specific to DetachedTripletStep
        Pt_min        = 0.9,
        DeltaR        = 2.0, # default = 0.2
        DeltaZ        = 2.0, # this give you the length
        Rescale_Dz    = 4.,  # max(DeltaZ_Region,Rescale_Dz*vtx->zError())
    )
)

###################################
import RecoTracker.IterativeTracking.DetachedTripletStep_cff

# NEW CLUSTERS (remove previously used clusters)
from RecoLocalTracker.SubCollectionProducers.trackClusterRemover_cfi import trackClusterRemover as _trackClusterRemover
hiRegitMuDetachedTripletStepClusters = _trackClusterRemover.clone(
    maxChi2                                  = 9.0,
    pixelClusters                            = "siPixelClusters",
    stripClusters                            = "siStripClusters",
    trajectories          		     = "hiRegitMuPixelLessStepTracks",
    overrideTrkQuals      		     = 'hiRegitMuPixelLessStepSelector:hiRegitMuPixelLessStep',
    TrackQuality                             = 'tight',
    trackClassifier       		     = '',
    minNumberOfLayersWithMeasBeforeFiltering = 0
)


# SEEDING LAYERS
hiRegitMuDetachedTripletStepSeedLayers =  RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepSeedLayers.clone(
    BPix = dict( skipClusters = 'hiRegitMuDetachedTripletStepClusters'),
    FPix = dict( skipClusters = 'hiRegitMuDetachedTripletStepClusters')
)

# seeding
hiRegitMuDetachedTripletStepHitDoublets = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepHitDoublets.clone(
    seedingLayers   = "hiRegitMuDetachedTripletStepSeedLayers",
    trackingRegions = "hiRegitMuDetachedTripletStepTrackingRegions",
    clusterCheck    = "hiRegitMuClusterCheck",
)

hiRegitMuDetachedTripletStepHitTriplets = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepHitTriplets.clone(
    doublets = "hiRegitMuDetachedTripletStepHitDoublets"
)

hiRegitMuDetachedTripletStepSeeds = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepSeeds.clone(
    seedingHitSets = "hiRegitMuDetachedTripletStepHitTriplets"
)
from RecoTracker.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *


# building: feed the new-named seeds
hiRegitMuDetachedTripletStepTrajectoryFilterBase = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepTrajectoryFilterBase.clone(
    minPt = 0.8 # after each new hit, apply pT cut for traj w/ at least minHitsMinPt = 3,
)

hiRegitMuDetachedTripletStepTrajectoryFilter = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepTrajectoryFilter.clone(
    filters = cms.VPSet(
      cms.PSet( refToPSet_ = cms.string('hiRegitMuDetachedTripletStepTrajectoryFilterBase')),
      cms.PSet( refToPSet_ = cms.string('detachedTripletStepTrajectoryFilterShape')))
)

hiRegitMuDetachedTripletStepTrajectoryBuilder = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepTrajectoryBuilder.clone(
    clustersToSkip = cms.InputTag('hiRegitMuDetachedTripletStepClusters')
)

hiRegitMuDetachedTripletStepTrackCandidates = RecoTracker.IterativeTracking.DetachedTripletStep_cff._detachedTripletStepTrackCandidatesCkf.clone(
    src               = 'hiRegitMuDetachedTripletStepSeeds',
    clustersToSkip    = 'hiRegitMuDetachedTripletStepClusters'
)

# fitting: feed new-names
hiRegitMuDetachedTripletStepTracks = RecoTracker.IterativeTracking.DetachedTripletStep_cff.detachedTripletStepTracks.clone(
    AlgorithmName = 'hiRegitMuDetachedTripletStep',
    src = 'hiRegitMuDetachedTripletStepTrackCandidates'
)

import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
import RecoHI.HiTracking.hiMultiTrackSelector_cfi
hiRegitMuDetachedTripletStepSelector = RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiMultiTrackSelector.clone(
    src            = 'hiRegitMuDetachedTripletStepTracks',
    vertices       = "hiSelectedPixelVertex",
    useAnyMVA      = True,
    GBRForestLabel = 'HIMVASelectorIter7',
    GBRForestVars  = ['chi2perdofperlayer', 'nhits', 'nlayers', 'eta'],
    trackSelectors = cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name      = 'hiRegitMuDetachedTripletStepLoose',
            min_nhits = 8
        ),
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
            name          = 'hiRegitMuDetachedTripletStepTight',
            preFilterName = 'hiRegitMuDetachedTripletStepLoose',
            min_nhits     = 8,
            useMVA        = True,
            minMVA        = -0.2
        ),
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
            name          = 'hiRegitMuDetachedTripletStep',
            preFilterName = 'hiRegitMuDetachedTripletStepTight',
            min_nhits     = 8,
            useMVA        = True,
            minMVA        = -0.09
        )
    ) #end of vpset
)
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
trackingPhase1.toModify(hiRegitMuDetachedTripletStepSelector, 
        useAnyMVA = False,
        trackSelectors= cms.VPSet(
            RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
                name      = 'hiRegitMuDetachedTripletStepLoose',
                min_nhits = 8
            ),
            RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
                name          = 'hiRegitMuDetachedTripletStepTight',
                preFilterName = 'hiRegitMuDetachedTripletStepLoose',
                min_nhits     = 8,
                useMVA        = False,
                minMVA        = -0.2
            ),
            RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
                name          = 'hiRegitMuDetachedTripletStep',
                preFilterName = 'hiRegitMuDetachedTripletStepTight',
                min_nhits     = 8,
                useMVA        = False,
                minMVA        = -0.09
            )
        )
)

hiRegitMuonDetachedTripletStepTask = cms.Task(hiRegitMuDetachedTripletStepClusters,
                                              hiRegitMuDetachedTripletStepSeedLayers,
                                              hiRegitMuDetachedTripletStepTrackingRegions,
                                              hiRegitMuDetachedTripletStepHitDoublets,
                                              hiRegitMuDetachedTripletStepHitTriplets,
                                              hiRegitMuDetachedTripletStepSeeds,
                                              hiRegitMuDetachedTripletStepTrackCandidates,
                                              hiRegitMuDetachedTripletStepTracks,
                                              hiRegitMuDetachedTripletStepSelector
                                              )
hiRegitMuonDetachedTripletStep = cms.Sequence(hiRegitMuonDetachedTripletStepTask)
