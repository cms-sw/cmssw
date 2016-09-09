import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras
import RecoTracker.IterativeTracking.iterativeTkConfig as _cfg

# NEW CLUSTERS (remove previously used clusters)
lowPtTripletStepClusters = _cfg.clusterRemoverForIter("LowPtTripletStep")
for era in _cfg.nonDefaultEras():
    getattr(eras, era).toReplaceWith(lowPtTripletStepClusters, _cfg.clusterRemoverForIter("LowPtTripletStep", era))

# SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi
lowPtTripletStepSeedLayers = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.PixelLayerTriplets.clone()
lowPtTripletStepSeedLayers.BPix.skipClusters = cms.InputTag('lowPtTripletStepClusters')
lowPtTripletStepSeedLayers.FPix.skipClusters = cms.InputTag('lowPtTripletStepClusters')
_layerListForPhase1 = [
    'BPix1+BPix2+BPix3', 'BPix2+BPix3+BPix4',
    'BPix1+BPix3+BPix4', 'BPix1+BPix2+BPix4',
    'BPix2+BPix3+FPix1_pos', 'BPix2+BPix3+FPix1_neg',
    'BPix1+BPix2+FPix1_pos', 'BPix1+BPix2+FPix1_neg',
    'BPix1+BPix3+FPix1_pos', 'BPix1+BPix3+FPix1_neg',
    'BPix2+FPix1_pos+FPix2_pos', 'BPix2+FPix1_neg+FPix2_neg',
    'BPix1+FPix1_pos+FPix2_pos', 'BPix1+FPix1_neg+FPix2_neg',
    'BPix1+BPix2+FPix2_pos', 'BPix1+BPix2+FPix2_neg',
    'FPix1_pos+FPix2_pos+FPix3_pos', 'FPix1_neg+FPix2_neg+FPix3_neg',
    'BPix1+FPix2_pos+FPix3_pos', 'BPix1+FPix2_neg+FPix3_neg',
    'BPix1+FPix1_pos+FPix3_pos', 'BPix1+FPix1_neg+FPix3_neg'
]
eras.trackingPhase1.toModify(lowPtTripletStepSeedLayers, layerList = _layerListForPhase1)
eras.trackingPhase1PU70.toModify(lowPtTripletStepSeedLayers, layerList = _layerListForPhase1)

_layerListForPhase2 = ['BPix1+BPix2+BPix3', 'BPix2+BPix3+BPix4',
                       'BPix1+BPix3+BPix4', 'BPix1+BPix2+BPix4',
                       'BPix1+BPix2+FPix1_pos', 'BPix1+BPix2+FPix1_neg',
                       'BPix1+FPix1_pos+FPix2_pos', 'BPix1+FPix1_neg+FPix2_neg',
                       'BPix1+BPix2+FPix2_pos', 'BPix1+BPix2+FPix2_neg',
                       'FPix1_pos+FPix2_pos+FPix3_pos', 'FPix1_neg+FPix2_neg+FPix3_neg',
                       'BPix1+FPix1_pos+FPix3_pos', 'BPix1+FPix1_neg+FPix3_neg',
                       'FPix2_pos+FPix3_pos+FPix4_pos', 'FPix2_neg+FPix3_neg+FPix4_neg',
                       'FPix3_pos+FPix4_pos+FPix5_pos', 'FPix3_neg+FPix4_neg+FPix5_neg',
                       'FPix4_pos+FPix5_pos+FPix6_pos', 'FPix4_neg+FPix5_neg+FPix6_neg',
                       'FPix5_pos+FPix6_pos+FPix7_pos', 'FPix5_neg+FPix6_neg+FPix7_neg',
                       'FPix6_pos+FPix7_pos+FPix8_pos', 'FPix6_neg+FPix7_neg+FPix8_neg'
]
eras.trackingPhase2PU140.toModify(lowPtTripletStepSeedLayers, layerList = _layerListForPhase2)

# SEEDS
import RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff
from RecoTracker.TkTrackingRegions.GlobalTrackingRegionFromBeamSpot_cfi import RegionPsetFomBeamSpotBlock
lowPtTripletStepSeeds = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone(
    RegionFactoryPSet = RegionPsetFomBeamSpotBlock.clone(
    ComponentName = cms.string('GlobalRegionProducerFromBeamSpot'),
    RegionPSet = RegionPsetFomBeamSpotBlock.RegionPSet.clone(
    ptMin = 0.2,
    originRadius = 0.02,
    nSigmaZ = 4.0
    )
    )
    )
lowPtTripletStepSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'lowPtTripletStepSeedLayers'
eras.trackingPhase1.toModify(lowPtTripletStepSeeds, # FIXME: Phase1PU70 value, let's see if we can lower it to Run2 value (0.2)
    RegionFactoryPSet = dict(RegionPSet = dict(ptMin = 0.35)),
)

eras.trackingPhase1PU70.toModify(lowPtTripletStepSeeds,
    RegionFactoryPSet = dict(
        RegionPSet = dict(
            ptMin = 0.35,
            originRadius = 0.015
        )
    ),
)
eras.trackingPhase2PU140.toModify(lowPtTripletStepSeeds,
     ClusterCheckPSet = dict(doClusterCheck = False),
     RegionFactoryPSet = dict(RegionPSet = dict(ptMin = 0.45)),
     OrderedHitsFactoryPSet = dict( GeneratorPSet = dict(maxElement = 0 ) ),
     SeedCreatorPSet = dict( magneticField = '', propagator = 'PropagatorWithMaterial'),
)

from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
import RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi
lowPtTripletStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.SeedComparitorPSet = RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi.LowPtClusterShapeSeedComparitor

eras.trackingPhase1.toModify(lowPtTripletStepSeeds,
    OrderedHitsFactoryPSet = dict(GeneratorPSet = dict(SeedComparitorPSet = dict(ComponentName = cms.string('none'))))
)

# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff as _TrajectoryFilter_cff
_lowPtTripletStepStandardTrajectoryFilterBase = _TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    minimumNumberOfHits = 3,
    minPt = 0.075,
)
lowPtTripletStepStandardTrajectoryFilter = _lowPtTripletStepStandardTrajectoryFilterBase.clone(
    maxCCCLostHits = 1,
    minGoodStripCharge = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutLoose'))
)
eras.trackingLowPU.toReplaceWith(lowPtTripletStepStandardTrajectoryFilter, _lowPtTripletStepStandardTrajectoryFilterBase)
eras.trackingPhase1PU70.toReplaceWith(lowPtTripletStepStandardTrajectoryFilter, _lowPtTripletStepStandardTrajectoryFilterBase)
eras.trackingPhase2PU140.toReplaceWith(lowPtTripletStepStandardTrajectoryFilter, _lowPtTripletStepStandardTrajectoryFilterBase)

from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeTrajectoryFilter_cfi import *
# Composite filter
lowPtTripletStepTrajectoryFilter = _TrajectoryFilter_cff.CompositeTrajectoryFilter_block.clone(
    filters   = [cms.PSet(refToPSet_ = cms.string('lowPtTripletStepStandardTrajectoryFilter')),
                 # cms.PSet(refToPSet_ = cms.string('ClusterShapeTrajectoryFilter'))
                ]
    )
eras.trackingPhase1PU70.toModify(lowPtTripletStepTrajectoryFilter,
    filters = lowPtTripletStepTrajectoryFilter.filters + [cms.PSet(refToPSet_ = cms.string('ClusterShapeTrajectoryFilter'))]
)
eras.trackingPhase2PU140.toModify(lowPtTripletStepTrajectoryFilter,
    filters = lowPtTripletStepTrajectoryFilter.filters + [cms.PSet(refToPSet_ = cms.string('ClusterShapeTrajectoryFilter'))]
)

import RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi
lowPtTripletStepChi2Est = RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi.Chi2ChargeMeasurementEstimator.clone(
    ComponentName = cms.string('lowPtTripletStepChi2Est'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(9.0),
    clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTiny')),
)
eras.trackingPhase1PU70.toModify(lowPtTripletStepChi2Est,
    clusterChargeCut = dict(refToPSet_ = 'SiStripClusterChargeCutNone'),
)

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
lowPtTripletStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    MeasurementTrackerName = '',
    trajectoryFilter = cms.PSet(refToPSet_ = cms.string('lowPtTripletStepTrajectoryFilter')),
    maxCand = 4,
    estimator = cms.string('lowPtTripletStepChi2Est'),
    maxDPhiForLooperReconstruction = cms.double(2.0),
    # 0.63 GeV is the maximum pT for a charged particle to loop within the 1.1m radius
    # of the outermost Tracker barrel layer (with B=3.8T)
    maxPtForLooperReconstruction = cms.double(0.7) 
    )
eras.trackingLowPU.toModify(lowPtTripletStepTrajectoryBuilder, maxCand = 3)
eras.trackingPhase2PU140.toModify(lowPtTripletStepTrajectoryBuilder, maxCand = 3)

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
lowPtTripletStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('lowPtTripletStepSeeds'),
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = cms.int32(50),
    onlyPixelHitsForSeedCleaner = cms.bool(True),

    TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string('lowPtTripletStepTrajectoryBuilder')),
    clustersToSkip = cms.InputTag('lowPtTripletStepClusters'),
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True
)
eras.trackingPhase2PU140.toModify(lowPtTripletStepTrackCandidates,
    clustersToSkip = None,
    phase2clustersToSkip = cms.InputTag("lowPtTripletStepClusters")
)

# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
lowPtTripletStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'lowPtTripletStepTrackCandidates',
    AlgorithmName = cms.string('lowPtTripletStep'),
    Fitter = cms.string('FlexibleKFFittingSmoother')
    )

from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi import trajectoryCleanerBySharedHits
lowPtTripletStepTrajectoryCleanerBySharedHits = trajectoryCleanerBySharedHits.clone(
        ComponentName = cms.string('lowPtTripletStepTrajectoryCleanerBySharedHits'),
            fractionShared = cms.double(0.16),
            allowSharedFirstHit = cms.bool(True)
            )
lowPtTripletStepTrackCandidates.TrajectoryCleaner = 'lowPtTripletStepTrajectoryCleanerBySharedHits'
eras.trackingLowPU.toModify(lowPtTripletStepTrajectoryCleanerBySharedHits, fractionShared = 0.19)
eras.trackingPhase1PU70.toModify(lowPtTripletStepTrajectoryCleanerBySharedHits, fractionShared = 0.09)
eras.trackingPhase2PU140.toModify(lowPtTripletStepTrajectoryCleanerBySharedHits, fractionShared = 0.09)

# Final selection



from RecoTracker.FinalTrackSelectors.TrackMVAClassifierPrompt_cfi import *
lowPtTripletStep =  TrackMVAClassifierPrompt.clone()
lowPtTripletStep.src = 'lowPtTripletStepTracks'
lowPtTripletStep.GBRForestLabel = 'MVASelectorIter1_13TeV'
lowPtTripletStep.qualityCuts = [-0.6,-0.3,-0.1]

# For Phase1
# MVA selection to be enabled after re-training, for time being we go with cut-based selector
from RecoTracker.FinalTrackSelectors.TrackCutClassifier_cfi import TrackCutClassifier as _TrackCutClassifier
eras.trackingPhase1.toReplaceWith(lowPtTripletStep, _TrackCutClassifier.clone(
    src = "lowPtTripletStepTracks",
    vertices = "firstStepPrimaryVertices",
    mva = dict (
        minPixelHits = [1,1,1],
        maxChi2 = [9999.,9999.,9999.],
        maxChi2n = [2.0,0.9,0.5],
        minLayers = [3,3,3],
        min3DLayers = [3,3,3],
        maxLostLayers = [2,2,2],
        dz_par = dict(
            dz_par1 = [0.7,0.6,0.45],
            dz_par2 = [0.5,0.4,0.4],
            dz_exp = [4,4,4],
        ),
        dr_par = dict(
            dr_par1 = [0.8,0.7,0.6],
            dr_par2 = [0.5,0.4,0.3],
            dr_exp = [4,4,4],
            d0err_par = [0.002,0.002,0.001],
        ),
    )
))

# For LowPU and Phase1PU70
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
lowPtTripletStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src = 'lowPtTripletStepTracks',
    useAnyMVA = cms.bool(False),
    GBRForestLabel = cms.string('MVASelectorIter1'),
    trackSelectors= [
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'lowPtTripletStepLoose',
        ), #end of pset
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'lowPtTripletStepTight',
            preFilterName = 'lowPtTripletStepLoose',
        ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'QualityMasks',
            preFilterName = 'lowPtTripletStepTight',
        ),
    ] #end of vpset
) #end of clone
eras.trackingPhase1PU70.toModify(lowPtTripletStepSelector,
    useAnyMVA = None,
    GBRForestLabel = None,
    trackSelectors = [
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'lowPtTripletStepLoose',
            chi2n_par = 2.0,
            res_par = ( 0.003, 0.002 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 2,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.8, 4.0 ),
            dz_par1 = ( 0.7, 4.0 ),
            d0_par2 = ( 0.5, 4.0 ),
            dz_par2 = ( 0.5, 4.0 )
        ), #end of pset
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'lowPtTripletStepTight',
            preFilterName = 'lowPtTripletStepLoose',
            chi2n_par = 0.9,
            res_par = ( 0.003, 0.002 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 2,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.7, 4.0 ),
            dz_par1 = ( 0.6, 4.0 ),
            d0_par2 = ( 0.4, 4.0 ),
            dz_par2 = ( 0.4, 4.0 )
        ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'lowPtTripletStep',
            preFilterName = 'lowPtTripletStepTight',
            chi2n_par = 0.5,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 2,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.6, 4.0 ),
            dz_par1 = ( 0.45, 4.0 ),
            d0_par2 = ( 0.3, 4.0 ),
            dz_par2 = ( 0.4, 4.0 )
        ),
    ] #end of vpset
) #end of clone

eras.trackingPhase2PU140.toModify(lowPtTripletStepSelector,
    useAnyMVA = None,
    GBRForestLabel = None,
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'lowPtTripletStepLoose',
            chi2n_par = 1.2,
            res_par = ( 0.003, 0.002 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 2,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.7, 4.0 ),
            dz_par1 = ( 0.6, 4.0 ),
            d0_par2 = ( 0.6, 4.0 ),
            dz_par2 = ( 0.6, 4.0 )
            ), #end of pset
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'lowPtTripletStepTight',
            preFilterName = 'lowPtTripletStepLoose',
            chi2n_par = 0.7,
            res_par = ( 0.003, 0.002 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 2,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.6, 4.0 ),
            dz_par1 = ( 0.5, 4.0 ),
            d0_par2 = ( 0.5, 4.0 ),
            dz_par2 = ( 0.5, 4.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'lowPtTripletStep',
            preFilterName = 'lowPtTripletStepTight',
            chi2n_par = 0.4,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 2,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.5, 4.0 ),
            dz_par1 = ( 0.4, 4.0 ),
            d0_par2 = ( 0.45, 4.0 ),
            dz_par2 = ( 0.45, 4.0 )
            ),
        ), #end of vpset
    vertices = "pixelVertices"
) #end of clone



# Final sequence
LowPtTripletStep = cms.Sequence(lowPtTripletStepClusters*
                                lowPtTripletStepSeedLayers*
                                lowPtTripletStepSeeds*
                                lowPtTripletStepTrackCandidates*
                                lowPtTripletStepTracks*
                                lowPtTripletStep)
_LowPtTripletStep_LowPU = LowPtTripletStep.copy()
_LowPtTripletStep_LowPU.replace(lowPtTripletStep, lowPtTripletStepSelector)
eras.trackingLowPU.toReplaceWith(LowPtTripletStep, _LowPtTripletStep_LowPU)
_LowPtTripletStep_Phase1PU70 = LowPtTripletStep.copy()
_LowPtTripletStep_Phase1PU70.replace(lowPtTripletStep, lowPtTripletStepSelector)
eras.trackingPhase1PU70.toReplaceWith(LowPtTripletStep, _LowPtTripletStep_Phase1PU70)
eras.trackingPhase2PU140.toReplaceWith(LowPtTripletStep, _LowPtTripletStep_Phase1PU70)
