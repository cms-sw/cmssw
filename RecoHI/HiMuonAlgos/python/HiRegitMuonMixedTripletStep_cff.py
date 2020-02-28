import FWCore.ParameterSet.Config as cms

################################################################################### 
# pp iterative tracking modified for hiOffline reco (the vertex is the one reconstructed in HI)
################################### 4th step: large impact parameter tracking using mixed-triplet seeding

from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *
# Are the following values set to the same in every iteration? If yes,
# why not making the change in HITrackingRegionProducer_cfi directly
# once for all?
hiRegitMuMixedTripletStepTrackingRegionsA = HiTrackingRegionFactoryFromSTAMuonsEDProducer.clone(
    MuonSrc = "standAloneMuons:UpdatedAtVtx", # this is the same as default, why repeat?
    MuonTrackingRegionBuilder = dict(
        vertexCollection = "hiSelectedPixelVertex",
        UseVertex     = True,
        Phi_fixed     = True,
        Eta_fixed     = True,
        DeltaPhi      = 0.3,
        DeltaEta      = 0.2,
        # Ok, the following ones are specific to MixedTripletStep
        Pt_min        = 1.3,
        DeltaR        = 0.5, # default = 0.2
        DeltaZ        = 0.5, # this give you the length
        Rescale_Dz    = 4., # max(DeltaZ_Region,Rescale_Dz*vtx->zError())
    )
)
hiRegitMuMixedTripletStepTrackingRegionsB = hiRegitMuMixedTripletStepTrackingRegionsA.clone(
    MuonTrackingRegionBuilder = dict(Pt_min = 1.5)
)

###################################
from RecoTracker.IterativeTracking.MixedTripletStep_cff import *

# NEW CLUSTERS (remove previously used clusters)
hiRegitMuMixedTripletStepClusters = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepClusters.clone(
    oldClusterRemovalInfo = cms.InputTag("hiRegitMuPixelPairStepClusters"),
    trajectories          = cms.InputTag("hiRegitMuPixelPairStepTracks"),
    overrideTrkQuals      = cms.InputTag('hiRegitMuPixelPairStepSelector','hiRegitMuPixelPairStep'),
    trackClassifier       = cms.InputTag(''),
    TrackQuality          = cms.string('tight')
)


# SEEDING LAYERS A
hiRegitMuMixedTripletStepSeedLayersA =  RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSeedLayersA.clone()
hiRegitMuMixedTripletStepSeedLayersA.BPix.skipClusters = cms.InputTag('hiRegitMuMixedTripletStepClusters')
hiRegitMuMixedTripletStepSeedLayersA.FPix.skipClusters = cms.InputTag('hiRegitMuMixedTripletStepClusters')
hiRegitMuMixedTripletStepSeedLayersA.TEC.skipClusters  = cms.InputTag('hiRegitMuMixedTripletStepClusters')

# SEEDS A
hiRegitMuMixedTripletStepHitDoubletsA = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepHitDoubletsA.clone(
    seedingLayers = "hiRegitMuMixedTripletStepSeedLayersA",
    trackingRegions = "hiRegitMuMixedTripletStepTrackingRegionsA",
    clusterCheck = "hiRegitMuClusterCheck",
)
hiRegitMuMixedTripletStepHitTripletsA = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepHitTripletsA.clone(
    doublets = "hiRegitMuMixedTripletStepHitDoubletsA"
)
hiRegitMuMixedTripletStepSeedsA     = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSeedsA.clone(
    seedingHitSets = "hiRegitMuMixedTripletStepHitTripletsA"
)

# SEEDING LAYERS B
hiRegitMuMixedTripletStepSeedLayersB =  RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSeedLayersB.clone()
hiRegitMuMixedTripletStepSeedLayersB.BPix.skipClusters = cms.InputTag('hiRegitMuMixedTripletStepClusters')
hiRegitMuMixedTripletStepSeedLayersB.TIB.skipClusters  = cms.InputTag('hiRegitMuMixedTripletStepClusters')


hiRegitMuMixedTripletStepHitDoubletsB = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepHitDoubletsB.clone(
    seedingLayers = "hiRegitMuMixedTripletStepSeedLayersB",
    trackingRegions = "hiRegitMuMixedTripletStepTrackingRegionsB",
    clusterCheck = "hiRegitMuClusterCheck",
)
hiRegitMuMixedTripletStepHitTripletsB = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepHitTripletsB.clone(
    doublets = "hiRegitMuMixedTripletStepHitDoubletsB"
)
hiRegitMuMixedTripletStepSeedsB     = RecoTracker.IterativeTracking.MixedTripletStep_cff.mixedTripletStepSeedsA.clone(
    seedingHitSets = "hiRegitMuMixedTripletStepHitTripletsB"
)

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
    AlgorithmName = cms.string('hiRegitMuMixedTripletStep'),
    src                 = 'hiRegitMuMixedTripletStepTrackCandidates',
)


# TRACK SELECTION AND QUALITY FLAG SETTING.
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
import RecoHI.HiTracking.hiMultiTrackSelector_cfi
hiRegitMuMixedTripletStepSelector = RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiMultiTrackSelector.clone(
    src                 = 'hiRegitMuMixedTripletStepTracks',
    vertices            = cms.InputTag("hiSelectedPixelVertex"),
    useAnyMVA = cms.bool(True),
    GBRForestLabel = cms.string('HIMVASelectorIter7'),
    GBRForestVars = cms.vstring(['chi2perdofperlayer', 'nhits', 'nlayers', 'eta']),
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
           name = 'hiRegitMuMixedTripletStepLoose',
           min_nhits = cms.uint32(8)
            ),
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
            name = 'hiRegitMuMixedTripletStepTight',
            preFilterName = 'hiRegitMuMixedTripletStepLoose',
            min_nhits = cms.uint32(8),
            useMVA = cms.bool(True),
            minMVA = cms.double(-0.2)
            ),
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
            name = 'hiRegitMuMixedTripletStep',
            preFilterName = 'hiRegitMuMixedTripletStepTight',
            min_nhits = cms.uint32(8),
            useMVA = cms.bool(True),
            minMVA = cms.double(-0.09)
            )
        ) #end of vpset
    ) #end of clone
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
trackingPhase1.toModify(hiRegitMuMixedTripletStepSelector, useAnyMVA = cms.bool(False))
trackingPhase1.toModify(hiRegitMuMixedTripletStepSelector, trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
           name = 'hiRegitMuMixedTripletStepLoose',
           min_nhits = cms.uint32(8)
            ),
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
            name = 'hiRegitMuMixedTripletStepTight',
            preFilterName = 'hiRegitMuMixedTripletStepLoose',
            min_nhits = cms.uint32(8),
            useMVA = cms.bool(False),
            minMVA = cms.double(-0.2)
            ),
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
            name = 'hiRegitMuMixedTripletStep',
            preFilterName = 'hiRegitMuMixedTripletStepTight',
            min_nhits = cms.uint32(8),
            useMVA = cms.bool(False),
            minMVA = cms.double(-0.09)
            )
        ) #end of vpset
)

hiRegitMuonMixedTripletStepTask = cms.Task(hiRegitMuMixedTripletStepClusters,
                                         hiRegitMuMixedTripletStepSeedLayersA,
                                         hiRegitMuMixedTripletStepTrackingRegionsA,
                                         hiRegitMuMixedTripletStepHitDoubletsA,
                                         hiRegitMuMixedTripletStepHitTripletsA,
                                         hiRegitMuMixedTripletStepSeedsA,
                                         hiRegitMuMixedTripletStepSeedLayersB,
                                         hiRegitMuMixedTripletStepTrackingRegionsB,
                                         hiRegitMuMixedTripletStepHitDoubletsB,
                                         hiRegitMuMixedTripletStepHitTripletsB,
                                         hiRegitMuMixedTripletStepSeedsB,
                                         hiRegitMuMixedTripletStepSeeds,
                                         hiRegitMuMixedTripletStepTrackCandidates,
                                         hiRegitMuMixedTripletStepTracks,
                                         hiRegitMuMixedTripletStepSelector)
hiRegitMuonMixedTripletStep = cms.Sequence(hiRegitMuonMixedTripletStepTask)
