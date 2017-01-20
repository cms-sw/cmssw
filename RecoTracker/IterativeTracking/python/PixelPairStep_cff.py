import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_tracker_apv_vfp30_2016_cff import tracker_apv_vfp30_2016 as _tracker_apv_vfp30_2016
import RecoTracker.IterativeTracking.iterativeTkConfig as _cfg

# NEW CLUSTERS (remove previously used clusters)
pixelPairStepClusters = _cfg.clusterRemoverForIter("PixelPairStep")
for _eraName, _postfix, _era in _cfg.nonDefaultEras():
    _era.toReplaceWith(pixelPairStepClusters, _cfg.clusterRemoverForIter("PixelPairStep", _eraName, _postfix))


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
from Configuration.Eras.Modifier_trackingPhase1PU70_cff import trackingPhase1PU70
trackingPhase1PU70.toModify(pixelPairStepSeedLayers,
    layerList = [
        'BPix1+BPix2', 'BPix1+BPix3', 'BPix2+BPix3',
        'BPix2+BPix4', 'BPix3+BPix4',
        'BPix1+FPix1_pos', 'BPix1+FPix1_neg',
        'BPix2+FPix1_pos', 'BPix2+FPix1_neg', 
        'FPix1_pos+FPix2_pos', 'FPix1_neg+FPix2_neg',
        'FPix2_pos+FPix3_pos', 'FPix2_neg+FPix3_neg'
    ]
)

_layerListForPhase2 = ['BPix1+BPix2', 'BPix1+BPix3', 'BPix2+BPix3',
                       'BPix2+BPix4', 'BPix3+BPix4',
                       'BPix1+FPix1_pos', 'BPix1+FPix1_neg',
                       'BPix2+FPix1_pos', 'BPix2+FPix1_neg',
                       'BPix1+FPix2_pos', 'BPix1+FPix2_neg',
                       'FPix1_pos+FPix2_pos', 'FPix1_neg+FPix2_neg',
                       'FPix2_pos+FPix3_pos', 'FPix2_neg+FPix3_neg',
                       'FPix3_pos+FPix4_pos', 'FPix3_neg+FPix4_neg',
                       'FPix4_pos+FPix5_pos', 'FPix4_neg+FPix5_neg',
                       'FPix5_pos+FPix6_pos', 'FPix5_neg+FPix6_neg'
]
from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
trackingPhase2PU140.toModify(pixelPairStepSeedLayers, 
    layerList = _layerListForPhase2,
    BPix = dict(
        useErrorsFromParam = cms.bool(True),
        hitErrorRPhi = cms.double(0.0027),
        hitErrorRZ = cms.double(0.006),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelPairs'),
    ),
    FPix = dict(
        useErrorsFromParam = cms.bool(True),
        hitErrorRPhi = cms.double(0.0051),
        hitErrorRZ = cms.double(0.0036),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelPairs'),
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
from Configuration.Eras.Modifier_trackingLowPU_cff import trackingLowPU
trackingLowPU.toModify(pixelPairStepSeeds,
    RegionFactoryPSet = dict(RegionPSet = dict(
        VertexCollection = 'pixelVertices',
        useMultipleScattering = False
    ))
)
trackingPhase1PU70.toModify(pixelPairStepSeeds,
    RegionFactoryPSet = dict(
        RegionPSet = dict(
            ptMin = 1.2,
            useMultipleScattering = False,
            VertexCollection = "pixelVertices",
        )
    ),
)
trackingPhase2PU140.toModify(pixelPairStepSeeds,
    RegionFactoryPSet = dict(
        RegionPSet = dict(
            ptMin = 1.3,
            useMultipleScattering = False,
            VertexCollection = "pixelVertices",
        )
    ),
    ClusterCheckPSet = dict(doClusterCheck = False),
    OrderedHitsFactoryPSet = dict( maxElement = 0 ),
    SeedCreatorPSet = dict( magneticField = '', propagator = 'PropagatorWithMaterial'),
)

# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
_pixelPairStepTrajectoryFilterBase = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    minimumNumberOfHits = 3,
    minPt = 0.1,
)
pixelPairStepTrajectoryFilterBase = _pixelPairStepTrajectoryFilterBase.clone(
    seedPairPenalty =0,
    maxCCCLostHits = 0,
    minGoodStripCharge = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutLoose'))
)
from Configuration.Eras.Modifier_tracker_apv_vfp30_2016_cff import tracker_apv_vfp30_2016
_tracker_apv_vfp30_2016.toModify(pixelPairStepTrajectoryFilterBase, maxCCCLostHits = 2)
trackingLowPU.toReplaceWith(pixelPairStepTrajectoryFilterBase, _pixelPairStepTrajectoryFilterBase)
trackingPhase1PU70.toReplaceWith(pixelPairStepTrajectoryFilterBase, _pixelPairStepTrajectoryFilterBase.clone(
    maxLostHitsFraction = 1./10.,
    constantValueForLostHitsFractionFilter = 0.801,
))
trackingPhase2PU140.toReplaceWith(pixelPairStepTrajectoryFilterBase, _pixelPairStepTrajectoryFilterBase.clone(
    maxLostHitsFraction = 1./10.,
    constantValueForLostHitsFractionFilter = 0.701,
))
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
    clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutLoose')),
    pTChargeCutThreshold = cms.double(15.)
)
_tracker_apv_vfp30_2016.toModify(pixelPairStepChi2Est,
    clusterChargeCut = dict(refToPSet_ = "SiStripClusterChargeCutTiny")
)
trackingLowPU.toModify(pixelPairStepChi2Est,
    clusterChargeCut = dict(refToPSet_ = 'SiStripClusterChargeCutTiny'),
)
trackingPhase1PU70.toModify(pixelPairStepChi2Est,
    MaxChi2 = 16.0,
    clusterChargeCut = dict(refToPSet_ = 'SiStripClusterChargeCutNone'),
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
trackingLowPU.toModify(pixelPairStepTrajectoryBuilder, maxCand = 2)

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
trackingPhase2PU140.toModify(pixelPairStepTrackCandidates,
    clustersToSkip = None,
    phase2clustersToSkip = cms.InputTag("pixelPairStepClusters"),
    TrajectoryCleaner = "pixelPairStepTrajectoryCleanerBySharedHits"
)

from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi import trajectoryCleanerBySharedHits as _trajectoryCleanerBySharedHits
pixelPairStepTrajectoryCleanerBySharedHits = _trajectoryCleanerBySharedHits.clone(
    ComponentName = 'pixelPairStepTrajectoryCleanerBySharedHits',
    fractionShared = 0.095,
    allowSharedFirstHit = True
)
trackingPhase1PU70.toModify(pixelPairStepTrackCandidates, TrajectoryCleaner = 'pixelPairStepTrajectoryCleanerBySharedHits')
trackingPhase2PU140.toModify(pixelPairStepTrajectoryCleanerBySharedHits, fractionShared = 0.09)

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

# For LowPU and Phase1PU70
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
trackingPhase1PU70.toModify(pixelPairStepSelector,
    useAnyMVA = None,
    GBRForestLabel = None,
    trackSelectors = [
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'pixelPairStepLoose',
            chi2n_par = 0.9,
            res_par = ( 0.003, 0.002 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 3,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.4, 4.0 ),
            dz_par1 = ( 0.4, 4.0 ),
            d0_par2 = ( 0.4, 4.0 ),
            dz_par2 = ( 0.4, 4.0 )
        ), #end of pset
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'pixelPairStepTight',
            preFilterName = 'pixelPairStepLoose',
            chi2n_par = 0.6,
            res_par = ( 0.003, 0.002 ),
            minNumberLayers = 4,
            maxNumberLostLayers = 2,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.3, 4.0 ),
            dz_par1 = ( 0.3, 4.0 ),
            d0_par2 = ( 0.3, 4.0 ),
            dz_par2 = ( 0.3, 4.0 )
        ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'pixelPairStep',
            preFilterName = 'pixelPairStepTight',
            chi2n_par = 0.4,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 4,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.2, 4.0 ),
            dz_par1 = ( 0.25, 4.0 ),
            d0_par2 = ( 0.25, 4.0 ),
            dz_par2 = ( 0.25, 4.0 )
        ),
    ] #end of vpset
) #end of clone

trackingPhase2PU140.toModify(pixelPairStepSelector,
    useAnyMVA = None,
    GBRForestLabel = None,
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'pixelPairStepLoose',
            chi2n_par = 0.7,
            res_par = ( 0.003, 0.002 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 2,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.4, 4.0 ),
            dz_par1 = ( 0.4, 4.0 ),
            d0_par2 = ( 0.6, 4.0 ),
            dz_par2 = ( 0.45, 4.0 )
            ), #end of pset
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'pixelPairStepTight',
            preFilterName = 'pixelPairStepLoose',
            chi2n_par = 0.6,
            res_par = ( 0.003, 0.002 ),
            minNumberLayers = 4,
            maxNumberLostLayers = 2,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.35, 4.0 ),
            dz_par1 = ( 0.35, 4.0 ),
            d0_par2 = ( 0.5, 4.0 ),
            dz_par2 = ( 0.4, 4.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'pixelPairStep',
            preFilterName = 'pixelPairStepTight',
            chi2n_par = 0.5,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 5,
            maxNumberLostLayers = 2,
            minNumber3DLayers = 4,
            d0_par1 = ( 0.3, 4.0 ),
            dz_par1 = ( 0.3, 4.0 ),
            d0_par2 = ( 0.45, 4.0 ),
            dz_par2 = ( 0.35, 4.0 )
            ),
        ), #end of vpset
    vertices = "pixelVertices"
) #end of clone


# Final sequence
PixelPairStep = cms.Sequence(pixelPairStepClusters*
                         pixelPairStepSeedLayers*
                         pixelPairStepSeeds*
                         pixelPairStepTrackCandidates*
                         pixelPairStepTracks*
                         pixelPairStep)
_PixelPairStep_LowPU_Phase1PU70 = PixelPairStep.copy()
_PixelPairStep_LowPU_Phase1PU70.replace(pixelPairStep, pixelPairStepSelector)
trackingLowPU.toReplaceWith(PixelPairStep, _PixelPairStep_LowPU_Phase1PU70)
trackingPhase1PU70.toReplaceWith(PixelPairStep, _PixelPairStep_LowPU_Phase1PU70)
trackingPhase2PU140.toReplaceWith(PixelPairStep, _PixelPairStep_LowPU_Phase1PU70)
