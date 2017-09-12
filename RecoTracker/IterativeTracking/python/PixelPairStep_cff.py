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
# layers covering the region not covered by quadruplets (so it is
# just acting as backup of triplets)
_layerListForPhase1 = [
    'BPix1+BPix2', 'BPix1+BPix3', 'BPix2+BPix3',
    'BPix1+FPix1_pos', 'BPix1+FPix1_neg',
    'BPix2+FPix1_pos', 'BPix2+FPix1_neg',
]
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
from Configuration.Eras.Modifier_trackingPhase1QuadProp_cff import trackingPhase1QuadProp
trackingPhase1.toModify(pixelPairStepSeedLayers, layerList = _layerListForPhase1)
trackingPhase1QuadProp.toModify(pixelPairStepSeedLayers, layerList = _layerListForPhase1)

# only layers covering the region not covered by quadruplets
# (so it is just acting as backup of triplets)
_layerListForPhase2 = [
        'BPix1+BPix2', 'BPix1+BPix3', 'BPix2+BPix3',
        'BPix1+FPix1_pos', 'BPix1+FPix1_neg',
        'BPix2+FPix1_pos', 'BPix2+FPix1_neg'
]
# modifing these errors seems to make no difference
from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
trackingPhase2PU140.toModify(pixelPairStepSeedLayers, 
    layerList = _layerListForPhase2,
    BPix = dict(
        useErrorsFromParam = cms.bool(True),
        hitErrorRPhi = cms.double(0.0016),
        hitErrorRZ = cms.double(0.0035),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelPairs'),
    ),
    FPix = dict(
        useErrorsFromParam = cms.bool(True),
        hitErrorRPhi = cms.double(0.0030),
        hitErrorRZ = cms.double(0.0020),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelPairs'),
    )
)

# TrackingRegion
from RecoTracker.TkTrackingRegions.globalTrackingRegionWithVertices_cff import globalTrackingRegionWithVertices as _globalTrackingRegionWithVertices
pixelPairStepTrackingRegions = _globalTrackingRegionWithVertices.clone(RegionPSet = dict(
    ptMin = 0.6,
    originRadius = 0.015,
    fixedError = 0.03,
    useMultipleScattering = True,
))
from Configuration.Eras.Modifier_trackingLowPU_cff import trackingLowPU
trackingLowPU.toModify(pixelPairStepTrackingRegions, RegionPSet=dict(useMultipleScattering=False))
_region_Phase1 = dict(
    useMultipleScattering = False,
    maxNVertices = 5,
)
trackingPhase1.toModify(pixelPairStepTrackingRegions, RegionPSet=_region_Phase1)
trackingPhase1QuadProp.toModify(pixelPairStepTrackingRegions, RegionPSet=_region_Phase1)
trackingPhase2PU140.toModify(pixelPairStepTrackingRegions, RegionPSet=_region_Phase1)

# SEEDS
from RecoTracker.TkHitPairs.hitPairEDProducer_cfi import hitPairEDProducer as _hitPairEDProducer
pixelPairStepHitDoublets = _hitPairEDProducer.clone(
    seedingLayers = "pixelPairStepSeedLayers",
    trackingRegions = "pixelPairStepTrackingRegions",
    produceSeedingHitSets = True,
)
from RecoTracker.TkSeedGenerator.seedCreatorFromRegionConsecutiveHitsEDProducer_cff import seedCreatorFromRegionConsecutiveHitsEDProducer as _seedCreatorFromRegionConsecutiveHitsEDProducer
pixelPairStepSeeds = _seedCreatorFromRegionConsecutiveHitsEDProducer.clone(
    seedingHitSets = "pixelPairStepHitDoublets",
    SeedComparitorPSet = dict(# FIXME: is this defined in any cfi that could be imported instead of copy-paste?
        ComponentName = 'PixelClusterShapeSeedComparitor',
        FilterAtHelixStage = cms.bool(True),
        FilterPixelHits = cms.bool(True),
        FilterStripHits = cms.bool(False),
        ClusterShapeHitFilterName = cms.string('ClusterShapeHitFilter'),
        ClusterShapeCacheSrc = cms.InputTag('siPixelClusterShapeCache'),
    )
)

# Clone for the phase1 recovery mode
pixelPairStepSeedsA = pixelPairStepSeeds.clone()

# Recovery for L2L3
pixelPairStepSeedLayersB = pixelPairStepSeedLayers.clone(
    layerList = [
        'BPix1+BPix4',
    ]
)
from RecoTracker.TkTrackingRegions.pointSeededTrackingRegion_cfi import pointSeededTrackingRegion as _pointSeededTrackingRegion
pixelPairStepTrackingRegionsB = _pointSeededTrackingRegion.clone(
    RegionPSet = dict(
        ptMin = 0.6,
        originRadius = 0.015,
        mode = "VerticesFixed",
        zErrorVetex = 0.03,
        vertexCollection = "firstStepPrimaryVertices",
        beamSpot = "offlineBeamSpot",
        maxNVertices = 5,
        maxNRegions = 5,
        whereToUseMeasurementTracker = "Never",
        deltaEta = 1.2,
        deltaPhi = 0.5,
        points = dict(
            eta = [0.0],
            phi = [3.0],
        )
    )
)
pixelPairStepHitDoubletsB = pixelPairStepHitDoublets.clone(
    seedingLayers = "pixelPairStepSeedLayersB",
    trackingRegions = "pixelPairStepTrackingRegionsB",
)
pixelPairStepSeedsB = pixelPairStepSeedsA.clone(seedingHitSets = "pixelPairStepHitDoubletsB")

# Merge
from RecoTracker.TkSeedGenerator.GlobalCombinedSeeds_cfi import globalCombinedSeeds as _globalCombinedSeeds
_pixelPairStepSeedsMerged = _globalCombinedSeeds.clone(
    seedCollections = ["pixelPairStepSeedsA", "pixelPairStepSeedsB"],
)
trackingPhase1.toReplaceWith(pixelPairStepSeeds, _pixelPairStepSeedsMerged)
trackingPhase1QuadProp.toReplaceWith(pixelPairStepSeeds, _pixelPairStepSeedsMerged)



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
trackingPhase1.toModify(pixelPairStepTrajectoryFilterBase, minimumNumberOfHits = 4)
trackingPhase1QuadProp.toModify(pixelPairStepTrajectoryFilterBase, minimumNumberOfHits = 4)
trackingPhase2PU140.toReplaceWith(pixelPairStepTrajectoryFilterBase, _pixelPairStepTrajectoryFilterBase.clone(
    minimumNumberOfHits = 4,
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
from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeTrajectoryFilter_cfi import *
trackingPhase2PU140.toModify(pixelPairStepTrajectoryFilter,
    filters = pixelPairStepTrajectoryFilter.filters + [cms.PSet(refToPSet_ = cms.string('ClusterShapeTrajectoryFilter'))]
)



pixelPairStepTrajectoryFilterInOut = pixelPairStepTrajectoryFilterBase.clone(
    minimumNumberOfHits = 4,
    seedExtension = 1,
    strictSeedExtension = False, # allow inactive
    pixelSeedExtension = False,
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
_seedExtension = dict(
    inOutTrajectoryFilter = dict(refToPSet_ = "pixelPairStepTrajectoryFilterInOut"),
    useSameTrajFilter = False,
)
trackingPhase1.toModify(pixelPairStepTrajectoryBuilder, **_seedExtension)
trackingPhase1QuadProp.toModify(pixelPairStepTrajectoryBuilder, **_seedExtension)
trackingPhase2PU140.toModify(pixelPairStepTrajectoryBuilder, **_seedExtension)




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

# For LowPU and Phase2PU140
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
    vertices = "firstStepPrimaryVertices"
) #end of clone


# Final sequence
PixelPairStep = cms.Sequence(pixelPairStepClusters*
                         pixelPairStepSeedLayers*
                         pixelPairStepTrackingRegions*
                         pixelPairStepHitDoublets*
                         pixelPairStepSeeds*
                         pixelPairStepTrackCandidates*
                         pixelPairStepTracks*
                         pixelPairStep)
_PixelPairStep_LowPU_Phase2PU140 = PixelPairStep.copy()
_PixelPairStep_LowPU_Phase2PU140.replace(pixelPairStep, pixelPairStepSelector)
trackingLowPU.toReplaceWith(PixelPairStep, _PixelPairStep_LowPU_Phase2PU140)
trackingPhase2PU140.toReplaceWith(PixelPairStep, _PixelPairStep_LowPU_Phase2PU140)
_PixelPairStep_Phase1 = PixelPairStep.copy()
_PixelPairStep_Phase1.replace(pixelPairStepSeeds,
                              pixelPairStepSeedsA *
                              pixelPairStepSeedLayersB*pixelPairStepTrackingRegionsB*pixelPairStepHitDoubletsB*pixelPairStepSeedsB*
                              pixelPairStepSeeds)
trackingPhase1.toReplaceWith(PixelPairStep, _PixelPairStep_Phase1)
trackingPhase1QuadProp.toReplaceWith(PixelPairStep, _PixelPairStep_Phase1)
