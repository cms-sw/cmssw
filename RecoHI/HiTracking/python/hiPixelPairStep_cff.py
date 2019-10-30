import FWCore.ParameterSet.Config as cms



# NEW CLUSTERS (remove previously used clusters)
hiPixelPairClusters = cms.EDProducer("HITrackClusterRemover",
                                     clusterLessSolution= cms.bool(True),
                                     oldClusterRemovalInfo = cms.InputTag("hiLowPtTripletStepClusters"),
                                     trajectories = cms.InputTag("hiLowPtTripletStepTracks"),
                                     overrideTrkQuals = cms.InputTag('hiLowPtTripletStepSelector','hiLowPtTripletStep'),
                                     TrackQuality = cms.string('highPurity'),
                                     minNumberOfLayersWithMeasBeforeFiltering = cms.int32(0),
                                     pixelClusters = cms.InputTag("siPixelClusters"),
                                     stripClusters = cms.InputTag("siStripClusters"),
                                     Common = cms.PSet(
    maxChi2 = cms.double(9.0),
    ),
                                     Strip = cms.PSet(
    maxChi2 = cms.double(9.0),
    #Yen-Jie's mod to preserve merged clusters
    maxSize = cms.uint32(2)
    )
                                     )


# SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelLayerPairs_cfi
hiPixelPairSeedLayers = RecoTracker.TkSeedingLayers.PixelLayerPairs_cfi.PixelLayerPairs.clone(
            layerList = cms.vstring('BPix1+BPix2', 'BPix1+BPix3', 'BPix2+BPix3',
                                    'BPix1+FPix1_pos', 'BPix1+FPix1_neg',
                                    'BPix2+FPix1_pos', 'BPix2+FPix1_neg',
                                    'FPix1_pos+FPix2_pos', 'FPix1_neg+FPix2_neg'),
            BPix = cms.PSet(
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelPairs'),
        HitProducer = cms.string('siPixelRecHits'),
        skipClusters = cms.InputTag('hiPixelPairClusters')
        ),
            FPix = cms.PSet(
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelPairs'),
        HitProducer = cms.string('siPixelRecHits'),
        skipClusters = cms.InputTag('hiPixelPairClusters')
        )
)
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
trackingPhase1.toModify(hiPixelPairSeedLayers,
	layerList = cms.vstring('BPix1+BPix4','BPix1+FPix1_pos','BPix1+FPix1_neg')  #only use first and fourth barrel layers or first barrel and first forward layer around area where BPIX2+3 are inactive
)

# SEEDS
import RecoTracker.TkSeedGenerator.GlobalSeedsFromPairsWithVertices_cff
hiPixelPairSeeds = RecoTracker.TkSeedGenerator.GlobalSeedsFromPairsWithVertices_cff.globalSeedsFromPairsWithVertices.clone()
hiPixelPairSeeds.RegionFactoryPSet.RegionPSet.VertexCollection=cms.InputTag("hiSelectedPixelVertex")
hiPixelPairSeeds.RegionFactoryPSet.RegionPSet.ptMin = 1.0
hiPixelPairSeeds.RegionFactoryPSet.RegionPSet.originRadius = 0.005
hiPixelPairSeeds.RegionFactoryPSet.RegionPSet.nSigmaZ = 4.0
# sigmaZVertex is only used when usedFixedError is True -Matt
hiPixelPairSeeds.RegionFactoryPSet.RegionPSet.sigmaZVertex = 4.0
hiPixelPairSeeds.RegionFactoryPSet.RegionPSet.useFixedError = cms.bool(False)
hiPixelPairSeeds.OrderedHitsFactoryPSet.SeedingLayers = cms.InputTag('hiPixelPairSeedLayers')
hiPixelPairSeeds.OrderedHitsFactoryPSet.maxElement = 5000000
hiPixelPairSeeds.ClusterCheckPSet.MaxNumberOfPixelClusters = 5000000
hiPixelPairSeeds.ClusterCheckPSet.MaxNumberOfCosmicClusters = 50000000
hiPixelPairSeeds.ClusterCheckPSet.MaxNumberOfCosmicClusters = 50000000
del hiPixelPairSeeds.ClusterCheckPSet.cut

hiPixelPairSeeds.SeedComparitorPSet = cms.PSet(
    ComponentName = cms.string('PixelClusterShapeSeedComparitor'),
    FilterAtHelixStage = cms.bool(True),
    FilterPixelHits = cms.bool(True),
    FilterStripHits = cms.bool(False),
    ClusterShapeHitFilterName = cms.string('ClusterShapeHitFilter'),
    ClusterShapeCacheSrc = cms.InputTag("siPixelClusterShapeCache")
    )

#rectangular tracking region around area missing BPIX2/3 in Phase 1
from RecoTracker.TkTrackingRegions.pointSeededTrackingRegion_cfi import pointSeededTrackingRegion as _pointSeededTrackingRegion
hiPixelPairStepTrackingRegionPhase1 = _pointSeededTrackingRegion.clone(
    RegionPSet = dict(
        ptMin = 0.9,
        originRadius = 0.005,
        mode = "VerticesSigma",
        nSigmaZVertex = 4.0,
        vertexCollection = "hiSelectedPixelVertex",
        beamSpot = "offlineBeamSpot",
        whereToUseMeasurementTracker = "Never",
        deltaEta = 1.8,
        deltaPhi = 0.5,
        points = dict(
            eta = [0.0],
            phi = [3.0],
        )
    )
)

from RecoTracker.TkHitPairs.hitPairEDProducer_cfi import hitPairEDProducer as _hitPairEDProducer
hiPixelPairStepHitDoubletsPhase1 = _hitPairEDProducer.clone(
    seedingLayers = "hiPixelPairSeedLayers",
    trackingRegions = "hiPixelPairStepTrackingRegionPhase1",
    clusterCheck = "",
    produceSeedingHitSets = cms.bool(True), 
)

from RecoTracker.TkSeedGenerator.seedCreatorFromRegionConsecutiveHitsEDProducer_cff import seedCreatorFromRegionConsecutiveHitsEDProducer as _seedCreatorFromRegionConsecutiveHitsEDProducer
hiPixelPairStepSeedsPhase1 = _seedCreatorFromRegionConsecutiveHitsEDProducer.clone(
    seedingHitSets = "hiPixelPairStepHitDoubletsPhase1",
    SeedComparitorPSet = dict(
        ComponentName = 'PixelClusterShapeSeedComparitor',
        FilterAtHelixStage = cms.bool(True),
        FilterPixelHits = cms.bool(True),
        FilterStripHits = cms.bool(False),
        ClusterShapeHitFilterName = cms.string('ClusterShapeHitFilter'),
        ClusterShapeCacheSrc = cms.InputTag('siPixelClusterShapeCache'),
    )
)

# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
hiPixelPairTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    #maxLostHits = 0,
    minimumNumberOfHits = 6,
    minPt = 1.0
    )

import TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi
hiPixelPairChi2Est = TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi.Chi2MeasurementEstimator.clone(
    ComponentName = cms.string('hiPixelPairChi2Est'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(9.0)
    )

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
hiPixelPairTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
        MeasurementTrackerName = '',
        trajectoryFilter = cms.PSet(refToPSet_ = cms.string('hiPixelPairTrajectoryFilter')),
        maxCand = 3,
        estimator = cms.string('hiPixelPairChi2Est'),
        maxDPhiForLooperReconstruction = cms.double(2.0),
        maxPtForLooperReconstruction = cms.double(0.7) 
        )

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
hiPixelPairTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('hiPixelPairSeeds'),
    clustersToSkip = cms.InputTag('hiPixelPairClusters'),
    TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string('hiPixelPairTrajectoryBuilder')),
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = cms.int32(50),
    onlyPixelHitsForSeedCleaner = cms.bool(True),

    )
trackingPhase1.toModify(hiPixelPairTrackCandidates,
    src = cms.InputTag('hiPixelPairStepSeedsPhase1')
)


# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
hiPixelPairGlobalPrimTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    AlgorithmName = cms.string('pixelPairStep'),
    src = 'hiPixelPairTrackCandidates',
    Fitter = cms.string('FlexibleKFFittingSmoother')
    )



# Final selection
import RecoHI.HiTracking.hiMultiTrackSelector_cfi
hiPixelPairStepSelector = RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiMultiTrackSelector.clone(
    src='hiPixelPairGlobalPrimTracks',
    useAnyMVA = cms.bool(True),
    GBRForestLabel = cms.string('HIMVASelectorIter6'),
    GBRForestVars = cms.vstring(['chi2perdofperlayer', 'dxyperdxyerror', 'dzperdzerror', 'nhits', 'nlayers', 'eta']),
    trackSelectors= cms.VPSet(
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiLooseMTS.clone(
    name = 'hiPixelPairStepLoose',
    useMVA = cms.bool(False)
    ), #end of pset
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
    name = 'hiPixelPairStepTight',
    preFilterName = 'hiPixelPairStepLoose',
    useMVA = cms.bool(True),
    minMVA = cms.double(-0.58)
    ),
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
    name = 'hiPixelPairStep',
    preFilterName = 'hiPixelPairStepTight',
    useMVA = cms.bool(True),
    minMVA = cms.double(0.77)
    ),
    ) #end of vpset
    ) #end of clone
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
trackingPhase1.toModify(hiPixelPairStepSelector, useAnyMVA = cms.bool(False))
trackingPhase1.toModify(hiPixelPairStepSelector, trackSelectors= cms.VPSet(
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiLooseMTS.clone(
    name = 'hiPixelPairStepLoose',
    useMVA = cms.bool(False)
    ), #end of pset
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
    name = 'hiPixelPairStepTight',
    preFilterName = 'hiPixelPairStepLoose',
    useMVA = cms.bool(False),
    minMVA = cms.double(-0.58)
    ),
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
    name = 'hiPixelPairStep',
    preFilterName = 'hiPixelPairStepTight',
    useMVA = cms.bool(False),
    minMVA = cms.double(0.77)
    ),
    ) #end of vpset
)



# Final sequence
hiPixelPairStepTask = cms.Task(hiPixelPairClusters,
                               hiPixelPairSeedLayers,
                               hiPixelPairSeeds,
                               hiPixelPairTrackCandidates,
                               hiPixelPairGlobalPrimTracks,
                               hiPixelPairStepSelector)
hiPixelPairStep = cms.Sequence(hiPixelPairStepTask)
hiPixelPairStep_Phase1 = hiPixelPairStepTask.copy()
hiPixelPairStep_Phase1.replace(hiPixelPairSeeds, cms.Task(hiPixelPairStepTrackingRegionPhase1,hiPixelPairStepHitDoubletsPhase1,hiPixelPairStepSeedsPhase1) )
trackingPhase1.toReplaceWith(hiPixelPairStepTask, hiPixelPairStep_Phase1)
