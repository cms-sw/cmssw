import FWCore.ParameterSet.Config as cms

# pp iterative tracking modified for hiOffline reco (the vertex is the one reconstructed in HI)
################################### 0st step:pixel-triplet seeding, high-pT;
from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *
# Are the following values set to the same in every iteration? If yes,
# why not making the change in HITrackingRegionProducer_cfi directly
# once for all?
hiRegitMuInitialStepTrackingRegions = HiTrackingRegionFactoryFromSTAMuonsEDProducer.clone(
    MuonSrc = "standAloneMuons:UpdatedAtVtx", # this is the same as default, why repeat?
    MuonTrackingRegionBuilder = dict(
        vertexCollection = "hiSelectedPixelVertex",
        UseVertex     = True,
        Phi_fixed     = True,
        Eta_fixed     = True,
        DeltaPhi      = 0.3,
        DeltaEta      = 0.2,
        # Ok, the following ones are specific to InitialStep
        Pt_min          = 3.0,
        DeltaR          = 1, # default = 0.2
        DeltaZ          = 1, # this give you the length
        Rescale_Dz      = 4., # max(DeltaZ_Region,Rescale_Dz*vtx->zError())
    )
)

###################################  
from RecoTracker.IterativeTracking.InitialStep_cff import *

# SEEDING LAYERS
hiRegitMuInitialStepSeedLayers =  RecoTracker.IterativeTracking.InitialStep_cff.initialStepSeedLayers.clone()

# seeding
hiRegitMuInitialStepHitDoublets = RecoTracker.IterativeTracking.InitialStep_cff.initialStepHitDoublets.clone(
    seedingLayers = "hiRegitMuInitialStepSeedLayers",
    trackingRegions = "hiRegitMuInitialStepTrackingRegions",
    clusterCheck = "hiRegitMuClusterCheck"
)
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
trackingPhase1.toModify(hiRegitMuInitialStepHitDoublets, layerPairs = [0])

hiRegitMuInitialStepHitTriplets = RecoTracker.IterativeTracking.InitialStep_cff.initialStepHitTriplets.clone(
    doublets = "hiRegitMuInitialStepHitDoublets"
)
hiRegitMuInitialStepSeeds = RecoTracker.IterativeTracking.InitialStep_cff.initialStepSeeds.clone(
    seedingHitSets = "hiRegitMuInitialStepHitTriplets"
)


# building: feed the new-named seeds
hiRegitMuInitialStepTrajectoryFilterBase = RecoTracker.IterativeTracking.InitialStep_cff.initialStepTrajectoryFilterBase.clone(
    minPt = 2.5 # after each new hit, apply pT cut for traj w/ at least minHitsMinPt = cms.int32(3),
)
hiRegitMuInitialStepTrajectoryFilter = RecoTracker.IterativeTracking.InitialStep_cff.initialStepTrajectoryFilter.clone(
    filters = cms.VPSet(
      cms.PSet( refToPSet_ = cms.string('hiRegitMuInitialStepTrajectoryFilterBase')),
      cms.PSet( refToPSet_ = cms.string('initialStepTrajectoryFilterShape')))
)

hiRegitMuInitialStepTrajectoryBuilder = RecoTracker.IterativeTracking.InitialStep_cff.initialStepTrajectoryBuilder.clone(
    trajectoryFilter = cms.PSet(
       refToPSet_ = cms.string('hiRegitMuInitialStepTrajectoryFilter')
       ),
)

# track candidates
hiRegitMuInitialStepTrackCandidates = RecoTracker.IterativeTracking.InitialStep_cff.initialStepTrackCandidates.clone(
    src               = 'hiRegitMuInitialStepSeeds',
    TrajectoryBuilderPSet = cms.PSet(
       refToPSet_ = cms.string('hiRegitMuInitialStepTrajectoryBuilder')
       ),
    maxNSeeds         = 1000000
)

# fitting: feed new-names
hiRegitMuInitialStepTracks = RecoTracker.IterativeTracking.InitialStep_cff.initialStepTracks.clone(
    AlgorithmName = 'hiRegitMuInitialStep',
    src           = 'hiRegitMuInitialStepTrackCandidates'
)


import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
import RecoHI.HiTracking.hiMultiTrackSelector_cfi
hiRegitMuInitialStepSelector = RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiMultiTrackSelector.clone(
    src            ='hiRegitMuInitialStepTracks',
    vertices       = "hiSelectedPixelVertex",
    useAnyMVA      = True,
    GBRForestLabel = 'HIMVASelectorIter4',
    GBRForestVars  = ['chi2perdofperlayer', 'dxyperdxyerror', 'dzperdzerror', 'nhits', 'nlayers', 'eta'],
    trackSelectors = cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
           name = 'hiRegitMuInitialStepLoose',
           min_nhits = 8
            ), #end of pset
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
            name = 'hiRegitMuInitialStepTight',
            preFilterName = 'hiRegitMuInitialStepLoose',
            min_nhits = 8,
            useMVA = True,
            minMVA = -0.38
            ),
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
            name = 'hiRegitMuInitialStep',
            preFilterName = 'hiRegitMuInitialStepTight',
            min_nhits = 8,
            useMVA = True,
            minMVA = -0.77
            ),
        ) #end of vpset
    )
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
trackingPhase1.toModify(hiRegitMuInitialStepSelector, useAnyMVA = False)
trackingPhase1.toModify(hiRegitMuInitialStepSelector, trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
           name = 'hiRegitMuInitialStepLoose',
           min_nhits = 8
            ), #end of pset
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
            name = 'hiRegitMuInitialStepTight',
            preFilterName = 'hiRegitMuInitialStepLoose',
            min_nhits = 8,
            useMVA = False,
            minMVA = -0.38
            ),
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
            name = 'hiRegitMuInitialStep',
            preFilterName = 'hiRegitMuInitialStepTight',
            min_nhits = 8,
            useMVA = False,
            minMVA = -0.77
            ),
        )
)

hiRegitMuonInitialStepTask = cms.Task(hiRegitMuInitialStepSeedLayers,
                                      hiRegitMuInitialStepTrackingRegions,
                                      hiRegitMuInitialStepHitDoublets,
                                      hiRegitMuInitialStepHitTriplets,
                                      hiRegitMuInitialStepSeeds,
                                      hiRegitMuInitialStepTrackCandidates,
                                      hiRegitMuInitialStepTracks,
                                      hiRegitMuInitialStepSelector)
hiRegitMuonInitialStep = cms.Sequence(hiRegitMuonInitialStepTask)
