import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras


# NEW CLUSTERS (remove previously used clusters)
from RecoLocalTracker.SubCollectionProducers.trackClusterRemover_cfi import trackClusterRemover as _trackClusterRemover
_pixelPairStepClustersBase = _trackClusterRemover.clone(
    maxChi2                                  = cms.double(9.0),
    trajectories                             = cms.InputTag("lowPtTripletStepTracks"),
    pixelClusters                            = cms.InputTag("siPixelClusters"),
    stripClusters                            = cms.InputTag("siStripClusters"),
    oldClusterRemovalInfo                    = cms.InputTag("lowPtTripletStepClusters"),
    TrackQuality                             = cms.string('highPurity'),
    minNumberOfLayersWithMeasBeforeFiltering = cms.int32(0),
)
pixelPairStepClusters = _pixelPairStepClustersBase.clone(
    trackClassifier                          = cms.InputTag('lowPtTripletStep',"QualityMasks"),
)
eras.trackingLowPU.toReplaceWith(pixelPairStepClusters, _pixelPairStepClustersBase.clone(
    overrideTrkQuals                         = "lowPtTripletStepSelector:QualityMasks",
))


# SEEDING LAYERS
pixelPairStepSeedLayers = cms.EDProducer("SeedingLayersEDProducer",
    layerList = cms.vstring('BPix1+BPix2', 'BPix1+BPix3', 'BPix2+BPix3', 
        'BPix1+FPix1_pos', 'BPix1+FPix1_neg', 
        'BPix2+FPix1_pos', 'BPix2+FPix1_neg', 
        'FPix1_pos+FPix2_pos', 'FPix1_neg+FPix2_neg'),
    BPix = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        HitProducer = cms.string('siPixelRecHits'),
        skipClusters = cms.InputTag('pixelPairStepClusters')
    ),
    FPix = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        HitProducer = cms.string('siPixelRecHits'),
        skipClusters = cms.InputTag('pixelPairStepClusters')
    )
)

# SEEDS
import RecoTracker.TkSeedGenerator.GlobalSeedsFromPairsWithVertices_cff
pixelPairStepSeeds = RecoTracker.TkSeedGenerator.GlobalSeedsFromPairsWithVertices_cff.globalSeedsFromPairsWithVertices.clone()
pixelPairStepSeeds.RegionFactoryPSet.RegionPSet.VertexCollection = cms.InputTag("firstStepPrimaryVertices")
pixelPairStepSeeds.RegionFactoryPSet.RegionPSet.ptMin = 0.6
pixelPairStepSeeds.RegionFactoryPSet.RegionPSet.originRadius = 0.015
pixelPairStepSeeds.RegionFactoryPSet.RegionPSet.fixedError = 0.03
pixelPairStepSeeds.RegionFactoryPSet.RegionPSet.useMultipleScattering = True
pixelPairStepSeeds.OrderedHitsFactoryPSet.SeedingLayers = cms.InputTag('pixelPairStepSeedLayers')

pixelPairStepSeeds.SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('PixelClusterShapeSeedComparitor'),
        FilterAtHelixStage = cms.bool(True),
        FilterPixelHits = cms.bool(True),
        FilterStripHits = cms.bool(False),
        ClusterShapeHitFilterName = cms.string('ClusterShapeHitFilter'),
        ClusterShapeCacheSrc = cms.InputTag('siPixelClusterShapeCache'),
    )
eras.trackingLowPU.toModify(pixelPairStepSeeds,
    RegionFactoryPSet = dict(RegionPSet = dict(
        VertexCollection = 'pixelVertices',
        useMultipleScattering = False
    ))
)

# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
_pixelPairStepTrajectoryFilterBase = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    minimumNumberOfHits = 3,
    minPt = 0.1,
    )
pixelPairStepTrajectoryFilterBase = _pixelPairStepTrajectoryFilterBase.clone(
    seedPairPenalty =0,
    maxCCCLostHits = 2,
    minGoodStripCharge = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutLoose'))
)
eras.trackingLowPU.toReplaceWith(pixelPairStepTrajectoryFilterBase, _pixelPairStepTrajectoryFilterBase)
import RecoPixelVertexing.PixelLowPtUtilities.StripSubClusterShapeTrajectoryFilter_cfi
pixelPairStepTrajectoryFilterShape = RecoPixelVertexing.PixelLowPtUtilities.StripSubClusterShapeTrajectoryFilter_cfi.StripSubClusterShapeTrajectoryFilterTIX12.clone()
pixelPairStepTrajectoryFilter = cms.PSet(
    ComponentType = cms.string('CompositeTrajectoryFilter'),
    filters = cms.VPSet(
        cms.PSet( refToPSet_ = cms.string('pixelPairStepTrajectoryFilterBase')),
    #    cms.PSet( refToPSet_ = cms.string('pixelPairStepTrajectoryFilterShape'))
    ),
)



import RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi
pixelPairStepChi2Est = RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi.Chi2ChargeMeasurementEstimator.clone(
    ComponentName = cms.string('pixelPairStepChi2Est'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(9.0),
    clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTiny')),
    pTChargeCutThreshold = cms.double(15.)
)
eras.trackingLowPU.toModify(pixelPairStepChi2Est,
    clusterChargeCut = dict(refToPSet_ = 'SiStripClusterChargeCutTiny'),
)
# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
pixelPairStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    MeasurementTrackerName = '',
    trajectoryFilter = cms.PSet(refToPSet_ = cms.string('pixelPairStepTrajectoryFilter')),
    maxCand = 3,
    estimator = cms.string('pixelPairStepChi2Est'),
    maxDPhiForLooperReconstruction = cms.double(2.0),
    maxPtForLooperReconstruction = cms.double(0.7) 
    )
eras.trackingLowPU.toModify(pixelPairStepTrajectoryBuilder, maxCand = 2)

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
pixelPairStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('pixelPairStepSeeds'),
    clustersToSkip = cms.InputTag('pixelPairStepClusters'),
    TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string('pixelPairStepTrajectoryBuilder')),
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = cms.int32(50),
    onlyPixelHitsForSeedCleaner = cms.bool(True),

)


# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
pixelPairStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    AlgorithmName = cms.string('pixelPairStep'),
    src = 'pixelPairStepTrackCandidates',
    Fitter = cms.string('FlexibleKFFittingSmoother')
    )

# Final selection
from RecoTracker.FinalTrackSelectors.TrackMVAClassifierPrompt_cfi import *
pixelPairStep =  TrackMVAClassifierPrompt.clone()
pixelPairStep.src = 'pixelPairStepTracks'
pixelPairStep.GBRForestLabel = 'MVASelectorIter2_13TeV'
pixelPairStep.qualityCuts = [-0.2,0.0,0.3]

# For LowPU
import RecoTracker.IterativeTracking.LowPtTripletStep_cff
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
pixelPairStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src='pixelPairStepTracks',
    useAnyMVA = cms.bool(True),
    GBRForestLabel = cms.string('MVASelectorIter2'),
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'pixelPairStepLoose',
        ), #end of pset
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'pixelPairStepTight',
            preFilterName = 'pixelPairStepLoose',
        ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'QualityMasks',
            preFilterName = 'pixelPairStepTight',
        ),
    ),
    vertices = cms.InputTag("pixelVertices")#end of vpset
) #end of clone


# Final sequence
PixelPairStep = cms.Sequence(pixelPairStepClusters*
                         pixelPairStepSeedLayers*
                         pixelPairStepSeeds*
                         pixelPairStepTrackCandidates*
                         pixelPairStepTracks*
                         pixelPairStep)
_PixelPairStep_LowPU = PixelPairStep.copy()
_PixelPairStep_LowPU.replace(pixelPairStep, pixelPairStepSelector)
eras.trackingLowPU.toReplaceWith(PixelPairStep, _PixelPairStep_LowPU)
