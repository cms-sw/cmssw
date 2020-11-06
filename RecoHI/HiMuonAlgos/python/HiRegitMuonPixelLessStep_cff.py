import FWCore.ParameterSet.Config as cms

# pp iterative tracking modified for hiOffline reco (the vertex is the one reconstructed in HI)
################################### 5th step: large impact parameter tracking using TIB/TID/TEC stereo layer seeding

from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *
# Are the following values set to the same in every iteration? If yes,
# why not making the change in HITrackingRegionProducer_cfi directly
# once for all?
hiRegitMuPixelLessStepTrackingRegions = HiTrackingRegionFactoryFromSTAMuonsEDProducer.clone(
    MuonSrc = "standAloneMuons:UpdatedAtVtx", # this is the same as default, why repeat?
    MuonTrackingRegionBuilder = dict(
        vertexCollection = "hiSelectedPixelVertex",
        UseVertex     = True,
        Phi_fixed     = True,
        Eta_fixed     = True,
        # Ok, the following ones are specific to PixelLessStep
        DeltaPhi      = 0.2,
        DeltaEta      = 0.1,
        Pt_min        = 2.0,
        DeltaR        = 0.2, # default = 0.2
        DeltaZ        = 0.2, # this give you the length
        Rescale_Dz    = 4.,  # max(DeltaZ_Region,Rescale_Dz*vtx->zError())
    )
)

###################################
from RecoTracker.IterativeTracking.PixelLessStep_cff import *

# remove previously used clusters
hiRegitMuPixelLessStepClusters = RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepClusters.clone(
    oldClusterRemovalInfo = "hiRegitMuMixedTripletStepClusters",
    trajectories          = "hiRegitMuMixedTripletStepTracks",
    overrideTrkQuals      = 'hiRegitMuMixedTripletStepSelector:hiRegitMuMixedTripletStep',
    trackClassifier       = '',
    TrackQuality          = 'tight'
)

# SEEDING LAYERS
hiRegitMuPixelLessStepSeedLayers =  RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepSeedLayers.clone(
    TIB  = dict(skipClusters = 'hiRegitMuPixelLessStepClusters'),
    TID  = dict(skipClusters = 'hiRegitMuPixelLessStepClusters'),
    TEC  = dict(skipClusters = 'hiRegitMuPixelLessStepClusters'),
    MTIB = dict(skipClusters = 'hiRegitMuPixelLessStepClusters'),
    MTID = dict(skipClusters = 'hiRegitMuPixelLessStepClusters'),
    MTEC = dict(skipClusters = 'hiRegitMuPixelLessStepClusters')
)

# seeding
hiRegitMuPixelLessStepHitDoublets = RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepHitDoublets.clone(
    seedingLayers   = "hiRegitMuPixelLessStepSeedLayers",
    trackingRegions = "hiRegitMuPixelLessStepTrackingRegions",
    clusterCheck    = "hiRegitMuClusterCheck",
)
hiRegitMuPixelLessStepHitTriplets = RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepHitTriplets.clone(
    doublets = "hiRegitMuPixelLessStepHitDoublets"
)
hiRegitMuPixelLessStepSeeds = RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepSeeds.clone(
    seedingHitSets = "hiRegitMuPixelLessStepHitTriplets"
)


# building: feed the new-named seeds
hiRegitMuPixelLessStepTrajectoryFilter = RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepTrajectoryFilter.clone(
    minPt                = 1.7,
    minimumNumberOfHits  = 6,
    minHitsMinPt         = 4
)
hiRegitMuPixelLessStepTrajectoryBuilder = RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepTrajectoryBuilder.clone(
    trajectoryFilter = cms.PSet(
       refToPSet_ = cms.string('hiRegitMuPixelLessStepTrajectoryFilter')
       ),
    minNrOfHitsForRebuild = 6 #change from default 4
)

hiRegitMuPixelLessStepTrackCandidates =  RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepTrackCandidates.clone(
    src               = 'hiRegitMuPixelLessStepSeeds',
    TrajectoryBuilderPSet = cms.PSet(
       refToPSet_ = cms.string('hiRegitMuPixelLessStepTrajectoryBuilder')
       ),
    clustersToSkip    = 'hiRegitMuPixelLessStepClusters',
    maxNSeeds         = 1000000
)

# fitting: feed new-names
hiRegitMuPixelLessStepTracks = RecoTracker.IterativeTracking.PixelLessStep_cff.pixelLessStepTracks.clone(
    AlgorithmName = 'hiRegitMuPixelLessStep',
    src           = 'hiRegitMuPixelLessStepTrackCandidates'
)

import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
import RecoHI.HiTracking.hiMultiTrackSelector_cfi
hiRegitMuPixelLessStepSelector = RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiMultiTrackSelector.clone(
    src            = 'hiRegitMuPixelLessStepTracks',
    vertices       = "hiSelectedPixelVertex",
    useAnyMVA      = True,
    GBRForestLabel = 'HIMVASelectorIter7',
    GBRForestVars  = ['chi2perdofperlayer', 'nhits', 'nlayers', 'eta'],
    trackSelectors = cms.VPSet(  
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
           name      = 'hiRegitMuPixelLessStepLoose',
           min_nhits = 8
            ),
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
            name          = 'hiRegitMuPixelLessStepTight',
            preFilterName = 'hiRegitMuPixelLessStepLoose',
            min_nhits     = 8,
            useMVA        = True,
            minMVA        = -0.2
            ),
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
            name          = 'hiRegitMuPixelLessStep',
            preFilterName = 'hiRegitMuPixelLessStepTight',
            min_nhits     = 8,
            useMVA        = True,
            minMVA        = -0.09
            ),
        ) #end of vpset
)
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
trackingPhase1.toModify(hiRegitMuPixelLessStepSelector, useAnyMVA = False)
trackingPhase1.toModify(hiRegitMuPixelLessStepSelector, trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
           name      = 'hiRegitMuPixelLessStepLoose',
           min_nhits = 8
            ),
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
            name          = 'hiRegitMuPixelLessStepTight',
            preFilterName = 'hiRegitMuPixelLessStepLoose',
            min_nhits     = 8,
            useMVA        = False,
            minMVA        = -0.2
            ),
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
            name          = 'hiRegitMuPixelLessStep',
            preFilterName = 'hiRegitMuPixelLessStepTight',
            min_nhits     = 8,
            useMVA        = False,
            minMVA        = -0.09
            ),
        ) #end of vpset
)

hiRegitMuonPixelLessStepTask = cms.Task(hiRegitMuPixelLessStepClusters,
                                        hiRegitMuPixelLessStepSeedLayers,
                                        hiRegitMuPixelLessStepTrackingRegions,
                                        hiRegitMuPixelLessStepHitDoublets,
                                        hiRegitMuPixelLessStepHitTriplets,
                                        hiRegitMuPixelLessStepSeeds,
                                        hiRegitMuPixelLessStepTrackCandidates,
                                        hiRegitMuPixelLessStepTracks,
                                        hiRegitMuPixelLessStepSelector)
hiRegitMuonPixelLessStep = cms.Sequence(hiRegitMuonPixelLessStepTask)

