import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras
import RecoTracker.IterativeTracking.iterativeTkConfig as _cfg

# NEW CLUSTERS (remove previously used clusters)
lowPtQuadStepClusters = _cfg.clusterRemoverForIter("LowPtQuadStep")
for era in _cfg.nonDefaultEras():
    getattr(eras, era).toReplaceWith(lowPtQuadStepClusters, _cfg.clusterRemoverForIter("LowPtQuadStep", era))


# SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi
import RecoPixelVertexing.PixelTriplets.quadrupletseedmerging_cff
lowPtQuadStepSeedLayers = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.PixelLayerTriplets.clone(
    BPix = dict(skipClusters = cms.InputTag('lowPtQuadStepClusters')),
    FPix = dict(skipClusters = cms.InputTag('lowPtQuadStepClusters'))
)
eras.trackingPhase1.toModify(lowPtQuadStepSeedLayers,
    layerList = RecoPixelVertexing.PixelTriplets.quadrupletseedmerging_cff.PixelSeedMergerQuadruplets.layerList.value()
)

# SEEDS
import RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff
from RecoTracker.TkTrackingRegions.GlobalTrackingRegionFromBeamSpot_cfi import RegionPsetFomBeamSpotBlock
from RecoPixelVertexing.PixelTriplets.quadrupletseedmerging_cff import *
from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
from RecoPixelVertexing.PixelTriplets.PixelQuadrupletGenerator_cfi import PixelQuadrupletGenerator as _PixelQuadrupletGenerator
import RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi as _LowPtClusterShapeSeedComparitor_cfi
lowPtQuadStepSeeds = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone(
    RegionFactoryPSet = RegionPsetFomBeamSpotBlock.clone(
        ComponentName = cms.string('GlobalRegionProducerFromBeamSpot'),
        RegionPSet = RegionPsetFomBeamSpotBlock.RegionPSet.clone(
            ptMin = 0.2,
            originRadius = 0.02,
            nSigmaZ = 4.0
        )
    ),
    OrderedHitsFactoryPSet = dict(
        SeedingLayers = 'lowPtQuadStepSeedLayers',
        GeneratorPSet = dict(
            SeedComparitorPSet = _LowPtClusterShapeSeedComparitor_cfi.LowPtClusterShapeSeedComparitor
        )
    ),
)
eras.trackingPhase1.toModify(lowPtQuadStepSeeds,
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string("CombinedHitQuadrupletGenerator"),
        GeneratorPSet = _PixelQuadrupletGenerator.clone(
            extraHitRZtolerance = lowPtQuadStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.extraHitRZtolerance,
            extraHitRPhitolerance = lowPtQuadStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.extraHitRPhitolerance,
            SeedComparitorPSet = cms.PSet( ComponentName = cms.string('none')),
            maxChi2 = dict(
                pt1    = 0.8  , pt2    = 2,
                value1 = 2000, value2 = 100,
                enabled = True,
            ),
            extraPhiTolerance = dict(
                pt1    = 0.3, pt2    = 1,
                value1 = 0.4, value2 = 0.05,
                enabled = True,
            ),
            useBendingCorrection = True,
            fitFastCircle = True,
            fitFastCircleChi2Cut = True,
        ),
        TripletGeneratorPSet = lowPtQuadStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.clone(
          SeedComparitorPSet = cms.PSet(ComponentName = cms.string('none'))
        ),
        SeedingLayers = lowPtQuadStepSeeds.OrderedHitsFactoryPSet.SeedingLayers,
    ),
)
eras.trackingPhase1PU70.toModify(lowPtQuadStepSeeds,
    SeedMergerPSet = cms.PSet(
        layerList = cms.PSet(refToPSet_ = cms.string("PixelSeedMergerQuadruplets")),
	addRemainingTriplets = cms.bool(False),
	mergeTriplets = cms.bool(True),
	ttrhBuilderLabel = cms.string('PixelTTRHBuilderWithoutAngle')
    )
)
eras.trackingPhase2PU140.toModify(lowPtQuadStepSeeds,
     ClusterCheckPSet = dict(doClusterCheck = False),
     RegionFactoryPSet = dict(RegionPSet = dict(ptMin = 0.35)),
     OrderedHitsFactoryPSet = dict( GeneratorPSet = dict(maxElement = 0 ) ),
     SeedCreatorPSet = dict( magneticField = '', propagator = 'PropagatorWithMaterial'),
     SeedMergerPSet = cms.PSet(
       layerList = cms.PSet(refToPSet_ = cms.string("PixelSeedMergerQuadruplets")),
       addRemainingTriplets = cms.bool(False),
       mergeTriplets = cms.bool(True),
       ttrhBuilderLabel = cms.string('PixelTTRHBuilderWithoutAngle')
     )
)


# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff as _TrajectoryFilter_cff
_lowPtQuadStepTrajectoryFilterBase = _TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    minimumNumberOfHits = 3,
    minPt = 0.075,
)
lowPtQuadStepTrajectoryFilterBase = _lowPtQuadStepTrajectoryFilterBase.clone(
    maxCCCLostHits = 1,
    minGoodStripCharge = dict(refToPSet_ = 'SiStripClusterChargeCutLoose')
)
eras.trackingPhase1PU70.toReplaceWith(lowPtQuadStepTrajectoryFilterBase, _lowPtQuadStepTrajectoryFilterBase)

from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeTrajectoryFilter_cfi import *
# Composite filter
lowPtQuadStepTrajectoryFilter = _TrajectoryFilter_cff.CompositeTrajectoryFilter_block.clone(
    filters = [cms.PSet(refToPSet_ = cms.string('lowPtQuadStepTrajectoryFilterBase'))]
)
eras.trackingPhase1PU70.toModify(lowPtQuadStepTrajectoryFilter,
    filters = lowPtQuadStepTrajectoryFilter.filters.value() + [cms.PSet(refToPSet_ = cms.string('ClusterShapeTrajectoryFilter'))]
)
eras.trackingPhase2PU140.toModify(lowPtQuadStepTrajectoryFilter,
    filters = lowPtQuadStepTrajectoryFilter.filters.value() + [cms.PSet(refToPSet_ = cms.string('ClusterShapeTrajectoryFilter'))]
)

import RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi
lowPtQuadStepChi2Est = RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi.Chi2ChargeMeasurementEstimator.clone(
    ComponentName = 'lowPtQuadStepChi2Est',
    nSigma = 3.0,
    MaxChi2 = 9.0,
    clusterChargeCut = dict(refToPSet_ = ('SiStripClusterChargeCutTiny')),
)
eras.trackingPhase1PU70.toModify(lowPtQuadStepChi2Est,
    MaxChi2 = 25.0,
    clusterChargeCut = dict(refToPSet_ = 'SiStripClusterChargeCutNone')
)
eras.trackingPhase2PU140.toModify(lowPtQuadStepChi2Est,
    MaxChi2 = 25.0,
    clusterChargeCut = dict(refToPSet_ = 'SiStripClusterChargeCutNone')
)

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
lowPtQuadStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    MeasurementTrackerName = '',
    trajectoryFilter = dict(refToPSet_ = 'lowPtQuadStepTrajectoryFilter'),
    maxCand = 4,
    estimator = cms.string('lowPtQuadStepChi2Est'),
    maxDPhiForLooperReconstruction = cms.double(2.0),
    # 0.63 GeV is the maximum pT for a charged particle to loop within the 1.1m radius
    # of the outermost Tracker barrel layer (with B=3.8T)
    maxPtForLooperReconstruction = cms.double(0.7) 
)
eras.trackingPhase2PU140.toModify(lowPtQuadStepTrajectoryBuilder, maxCand = 5)

# MAKING OF TRACK CANDIDATES
from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi import trajectoryCleanerBySharedHits as _trajectoryCleanerBySharedHits
lowPtQuadStepTrajectoryCleanerBySharedHits = _trajectoryCleanerBySharedHits.clone(
    ComponentName = 'lowPtQuadStepTrajectoryCleanerBySharedHits',
    fractionShared = 0.16,
    allowSharedFirstHit = True
)
eras.trackingPhase1PU70.toModify(lowPtQuadStepTrajectoryCleanerBySharedHits, fractionShared = 0.095)
eras.trackingPhase2PU140.toModify(lowPtQuadStepTrajectoryCleanerBySharedHits, fractionShared = 0.09)

import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
lowPtQuadStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = 'lowPtQuadStepSeeds',
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = cms.int32(50),
    onlyPixelHitsForSeedCleaner = cms.bool(True),
    TrajectoryBuilderPSet = dict(refToPSet_ = 'lowPtQuadStepTrajectoryBuilder'),
    TrajectoryCleaner = 'lowPtQuadStepTrajectoryCleanerBySharedHits',
    clustersToSkip = cms.InputTag('lowPtQuadStepClusters'),
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True
)
eras.trackingPhase2PU140.toModify(lowPtQuadStepTrackCandidates,
    clustersToSkip = None,
    phase2clustersToSkip = cms.InputTag("lowPtQuadStepClusters")
)
# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
lowPtQuadStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'lowPtQuadStepTrackCandidates',
    AlgorithmName = 'lowPtQuadStep',
    Fitter = 'FlexibleKFFittingSmoother',
)



# Final selection
from RecoTracker.FinalTrackSelectors.TrackMVAClassifierPrompt_cfi import *
lowPtQuadStep =  TrackMVAClassifierPrompt.clone(
    src = 'lowPtQuadStepTracks',
    GBRForestLabel = 'MVASelectorIter1_13TeV',
    qualityCuts = [-0.6,-0.3,-0.1]
)
# For Phase1PU70
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
lowPtQuadStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src = 'lowPtQuadStepTracks',
    trackSelectors = [
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'lowPtQuadStepLoose',
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
            name = 'lowPtQuadStepTight',
            preFilterName = 'lowPtQuadStepLoose',
            chi2n_par = 1.3,
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
            name = 'lowPtQuadStep',
            preFilterName = 'lowPtQuadStepTight',
            chi2n_par = 1.0,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 2,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.6, 4.0 ),
            dz_par1 = ( 0.5, 4.0 ),
            d0_par2 = ( 0.3, 4.0 ),
            dz_par2 = ( 0.4, 4.0 )
        ),
    ]
) #end of clone

eras.trackingPhase2PU140.toModify(lowPtQuadStepSelector,
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'lowPtQuadStepLoose',
            chi2n_par = 2.0,
            res_par = ( 0.003, 0.002 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 2,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.8, 4.0 ),
            dz_par1 = ( 0.7, 4.0 ),
            d0_par2 = ( 0.6, 4.0 ),
            dz_par2 = ( 0.6, 4.0 )
            ), #end of pset
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'lowPtQuadStepTight',
            preFilterName = 'lowPtQuadStepLoose',
            chi2n_par = 1.4,
            res_par = ( 0.003, 0.002 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 2,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.7, 4.0 ),
            dz_par1 = ( 0.6, 4.0 ),
            d0_par2 = ( 0.5, 4.0 ),
            dz_par2 = ( 0.5, 4.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'lowPtQuadStep',
            preFilterName = 'lowPtQuadStepTight',
            chi2n_par = 1.2,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 2,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.5, 4.0 ),
            dz_par1 = ( 0.5, 4.0 ),
            d0_par2 = ( 0.45, 4.0 ),
            dz_par2 = ( 0.45, 4.0 )
            ),
        ), #end of vpset
    vertices = "pixelVertices"
) #end of clone

# Final sequence
LowPtQuadStep = cms.Sequence(lowPtQuadStepClusters*
                             lowPtQuadStepSeedLayers*
                             lowPtQuadStepSeeds*
                             lowPtQuadStepTrackCandidates*
                             lowPtQuadStepTracks*
                             lowPtQuadStep)
_LowPtQuadStep_Phase1PU70 = LowPtQuadStep.copy()
_LowPtQuadStep_Phase1PU70.replace(lowPtQuadStep, lowPtQuadStepSelector)
eras.trackingPhase1PU70.toReplaceWith(LowPtQuadStep, _LowPtQuadStep_Phase1PU70)
eras.trackingPhase2PU140.toReplaceWith(LowPtQuadStep, _LowPtQuadStep_Phase1PU70)
