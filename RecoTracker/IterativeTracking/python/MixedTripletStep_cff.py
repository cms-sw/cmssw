import FWCore.ParameterSet.Config as cms
import RecoTracker.IterativeTracking.iterativeTkConfig as _cfg

from Configuration.Eras.Modifier_fastSim_cff import fastSim

#for dnn classifier
from Configuration.ProcessModifiers.trackdnn_cff import trackdnn

###############################################################
# Large impact parameter Tracking using mixed-triplet seeding #
###############################################################

#here just for backward compatibility
chargeCut2069Clusters =  cms.EDProducer("ClusterChargeMasker",
    oldClusterRemovalInfo = cms.InputTag(""), # to be set below
    pixelClusters = cms.InputTag("siPixelClusters"),
    stripClusters = cms.InputTag("siStripClusters"),
    clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight'))
)

mixedTripletStepClusters = _cfg.clusterRemoverForIter("MixedTripletStep")
chargeCut2069Clusters.oldClusterRemovalInfo = mixedTripletStepClusters.oldClusterRemovalInfo.value()
mixedTripletStepClusters.oldClusterRemovalInfo = "chargeCut2069Clusters"
for _eraName, _postfix, _era in _cfg.nonDefaultEras():
    _era.toReplaceWith(mixedTripletStepClusters, _cfg.clusterRemoverForIter("MixedTripletStep", _eraName, _postfix))
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
trackingPhase1.toModify(chargeCut2069Clusters, oldClusterRemovalInfo = mixedTripletStepClusters.oldClusterRemovalInfo.value())
trackingPhase1.toModify(mixedTripletStepClusters, oldClusterRemovalInfo="chargeCut2069Clusters")

# SEEDING LAYERS
from RecoLocalTracker.SiStripClusterizer.SiStripClusterChargeCut_cfi import *
from RecoTracker.IterativeTracking.DetachedTripletStep_cff import detachedTripletStepSeedLayers
mixedTripletStepSeedLayersA = cms.EDProducer("SeedingLayersEDProducer",
     layerList = cms.vstring('BPix2+FPix1_pos+FPix2_pos', 'BPix2+FPix1_neg+FPix2_neg'),
#    layerList = cms.vstring('BPix1+BPix2+BPix3', 
#        'BPix1+BPix2+FPix1_pos', 'BPix1+BPix2+FPix1_neg', 
#        'BPix1+FPix1_pos+FPix2_pos', 'BPix1+FPix1_neg+FPix2_neg', 
#        'BPix2+FPix1_pos+FPix2_pos', 'BPix2+FPix1_neg+FPix2_neg'),
    BPix = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        HitProducer = cms.string('siPixelRecHits'),
        skipClusters = cms.InputTag('mixedTripletStepClusters')
    ),
    FPix = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        HitProducer = cms.string('siPixelRecHits'),
        skipClusters = cms.InputTag('mixedTripletStepClusters')
    ),
    TEC = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'), clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight')),
        minRing = cms.int32(1),
        maxRing = cms.int32(1),
        skipClusters = cms.InputTag('mixedTripletStepClusters')
    )
)
from Configuration.Eras.Modifier_trackingLowPU_cff import trackingLowPU
trackingLowPU.toModify(mixedTripletStepSeedLayersA,
    layerList = [
        'BPix1+BPix2+BPix3',
        'BPix1+BPix2+FPix1_pos', 'BPix1+BPix2+FPix1_neg',
        'BPix1+FPix1_pos+FPix2_pos', 'BPix1+FPix1_neg+FPix2_neg',
        'BPix2+FPix1_pos+FPix2_pos', 'BPix2+FPix1_neg+FPix2_neg',
        'FPix1_pos+FPix2_pos+TEC1_pos', 'FPix1_neg+FPix2_neg+TEC1_neg',
        'FPix2_pos+TEC2_pos+TEC3_pos', 'FPix2_neg+TEC2_neg+TEC3_neg'
    ],
    TEC = dict(clusterChargeCut = dict(refToPSet_ = 'SiStripClusterChargeCutTiny')),
)
from Configuration.Eras.Modifier_highBetaStar_2018_cff import highBetaStar_2018
highBetaStar_2018.toModify(mixedTripletStepSeedLayersA,
    layerList = [
        'BPix1+BPix2+BPix3',
        'BPix1+FPix1_pos+FPix2_pos','BPix1+FPix1_neg+FPix2_neg',
        'BPix2+FPix1_pos+FPix2_pos','BPix2+FPix1_neg+FPix2_neg',
        'BPix1+BPix2+FPix1_pos', 'BPix1+BPix2+FPix1_neg',
        'BPix1+BPix2+FPix2_pos', 'BPix1+BPix2+FPix2_neg'
    ]
)


# TrackingRegion
from RecoTracker.TkTrackingRegions.globalTrackingRegionFromBeamSpotFixedZ_cfi import globalTrackingRegionFromBeamSpotFixedZ as _globalTrackingRegionFromBeamSpotFixedZ
_mixedTripletStepTrackingRegionsCommon = _globalTrackingRegionFromBeamSpotFixedZ.clone(RegionPSet = dict(
    ptMin = 0.4,
    originHalfLength = 15.0,
    originRadius = 1.5
))
trackingLowPU.toModify(_mixedTripletStepTrackingRegionsCommon, RegionPSet = dict(originHalfLength = 10.0))
highBetaStar_2018.toModify(_mixedTripletStepTrackingRegionsCommon,RegionPSet = dict(
     ptMin = 0.05,
     originRadius = 0.2
))

mixedTripletStepTrackingRegionsA = _mixedTripletStepTrackingRegionsCommon.clone()

from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
from RecoTracker.TkTrackingRegions.globalTrackingRegionWithVertices_cff import globalTrackingRegionWithVertices as _globalTrackingRegionWithVertices
_mixedTripletStepTrackingRegionsCommon_pp_on_HI = _globalTrackingRegionWithVertices.clone(
                RegionPSet=dict(
                    fixedError = 3.75,
                    ptMin = 0.4,
                    originRadius = 1.5,
                    originRScaling4BigEvts = True,
                    ptMinScaling4BigEvts= True,
                    minOriginR = 0.,
                    maxPtMin = 0.7,
                    scalingStartNPix = 20000,
                    scalingEndNPix = 35000
                )
)
(pp_on_XeXe_2017 | pp_on_AA_2018).toReplaceWith(mixedTripletStepTrackingRegionsA,_mixedTripletStepTrackingRegionsCommon_pp_on_HI)


# seeding
from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import ClusterShapeHitFilterESProducer as _ClusterShapeHitFilterESProducer
mixedTripletStepClusterShapeHitFilter  = _ClusterShapeHitFilterESProducer.clone(
    ComponentName = 'mixedTripletStepClusterShapeHitFilter',
    clusterChargeCut = dict(refToPSet_ = 'SiStripClusterChargeCutTight')
)
from RecoTracker.TkHitPairs.hitPairEDProducer_cfi import hitPairEDProducer as _hitPairEDProducer
mixedTripletStepHitDoubletsA = _hitPairEDProducer.clone(
    seedingLayers = "mixedTripletStepSeedLayersA",
    trackingRegions = "mixedTripletStepTrackingRegionsA",
    maxElement = 50000000,
    produceIntermediateHitDoublets = True,
)
from RecoPixelVertexing.PixelTriplets.pixelTripletLargeTipEDProducer_cfi import pixelTripletLargeTipEDProducer as _pixelTripletLargeTipEDProducer
from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
mixedTripletStepHitTripletsA = _pixelTripletLargeTipEDProducer.clone(
    doublets = "mixedTripletStepHitDoubletsA",
    produceSeedingHitSets = True,
)
from RecoTracker.TkSeedGenerator.seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer_cff import seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer as _seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer
_mixedTripletStepSeedsACommon = _seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer.clone(
    seedingHitSets = "mixedTripletStepHitTripletsA",
    SeedComparitorPSet = dict(# FIXME: is this defined in any cfi that could be imported instead of copy-paste?
        ComponentName = 'PixelClusterShapeSeedComparitor',
        FilterAtHelixStage = cms.bool(False),
        FilterPixelHits = cms.bool(True),
        FilterStripHits = cms.bool(True),
        ClusterShapeHitFilterName = cms.string('mixedTripletStepClusterShapeHitFilter'),
        ClusterShapeCacheSrc = cms.InputTag('siPixelClusterShapeCache')
    ),
)
trackingLowPU.toModify(_mixedTripletStepSeedsACommon,
    SeedComparitorPSet = dict(ClusterShapeHitFilterName = 'ClusterShapeHitFilter')
)
mixedTripletStepSeedsA = _mixedTripletStepSeedsACommon.clone()

import FastSimulation.Tracking.TrajectorySeedProducer_cfi
from FastSimulation.Tracking.SeedingMigration import _hitSetProducerToFactoryPSet
_fastSim_mixedTripletStepSeedsA = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    trackingRegions = "mixedTripletStepTrackingRegionsA",
    hitMasks = cms.InputTag("mixedTripletStepMasks"),
    seedFinderSelector = dict(pixelTripletGeneratorFactory = _hitSetProducerToFactoryPSet(mixedTripletStepHitTripletsA),
                              layerList = mixedTripletStepSeedLayersA.layerList.value())
)
fastSim.toReplaceWith(mixedTripletStepSeedsA,_fastSim_mixedTripletStepSeedsA)


# SEEDING LAYERS
mixedTripletStepSeedLayersB = cms.EDProducer("SeedingLayersEDProducer",
    layerList = cms.vstring('BPix2+BPix3+TIB1'),
    BPix = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        HitProducer = cms.string('siPixelRecHits'),
        skipClusters = cms.InputTag('mixedTripletStepClusters')
    ),
    TIB = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle'), clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight')),
        skipClusters = cms.InputTag('mixedTripletStepClusters')
    )
)
trackingLowPU.toModify(mixedTripletStepSeedLayersB,
    layerList = ['BPix2+BPix3+TIB1', 'BPix2+BPix3+TIB2'],
    TIB = dict(clusterChargeCut = dict(refToPSet_ = 'SiStripClusterChargeCutTiny')),
)
trackingPhase1.toModify(mixedTripletStepSeedLayersB, layerList = ['BPix3+BPix4+TIB1'])

# TrackingRegion
mixedTripletStepTrackingRegionsB = _mixedTripletStepTrackingRegionsCommon.clone(RegionPSet = dict(ptMin=0.6, originHalfLength=10.0))
(pp_on_XeXe_2017 | pp_on_AA_2018).toReplaceWith(mixedTripletStepTrackingRegionsB, 
                _mixedTripletStepTrackingRegionsCommon_pp_on_HI.clone(RegionPSet=dict(
                    fixedError = 2.5,
                    ptMin = 0.6,
                )
           )
)
highBetaStar_2018.toReplaceWith(mixedTripletStepTrackingRegionsB, _mixedTripletStepTrackingRegionsCommon.clone())

# seeding
mixedTripletStepHitDoubletsB = mixedTripletStepHitDoubletsA.clone(
    seedingLayers = "mixedTripletStepSeedLayersB",
    trackingRegions = "mixedTripletStepTrackingRegionsB",
)
mixedTripletStepHitTripletsB = mixedTripletStepHitTripletsA.clone(doublets = "mixedTripletStepHitDoubletsB")
mixedTripletStepSeedsB = _mixedTripletStepSeedsACommon.clone(seedingHitSets = "mixedTripletStepHitTripletsB")
#fastsim
_fastSim_mixedTripletStepSeedsB = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    trackingRegions = "mixedTripletStepTrackingRegionsB",
    hitMasks = cms.InputTag("mixedTripletStepMasks"),
    seedFinderSelector = dict(pixelTripletGeneratorFactory = _hitSetProducerToFactoryPSet(mixedTripletStepHitTripletsB),
                              layerList = mixedTripletStepSeedLayersB.layerList.value())
)
fastSim.toReplaceWith(mixedTripletStepSeedsB,_fastSim_mixedTripletStepSeedsB)


import RecoTracker.TkSeedGenerator.GlobalCombinedSeeds_cfi
mixedTripletStepSeeds = RecoTracker.TkSeedGenerator.GlobalCombinedSeeds_cfi.globalCombinedSeeds.clone()
mixedTripletStepSeeds.seedCollections = cms.VInputTag(
        cms.InputTag('mixedTripletStepSeedsA'),
        cms.InputTag('mixedTripletStepSeedsB'),
        )

# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
_mixedTripletStepTrajectoryFilterBase = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
#    maxLostHits = 0,
    minimumNumberOfHits = 3,
    minPt = 0.1
)
highBetaStar_2018.toModify(_mixedTripletStepTrajectoryFilterBase,minPt = 0.05)

mixedTripletStepTrajectoryFilter = _mixedTripletStepTrajectoryFilterBase.clone(
    constantValueForLostHitsFractionFilter = 1.4,
)
trackingLowPU.toReplaceWith(mixedTripletStepTrajectoryFilter, _mixedTripletStepTrajectoryFilterBase.clone(
    maxLostHits = 0,
))

for e in [pp_on_XeXe_2017, pp_on_AA_2018]:
    e.toModify(mixedTripletStepTrajectoryFilter, minPt=0.4)

# Propagator taking into account momentum uncertainty in multiple scattering calculation.
import TrackingTools.MaterialEffects.MaterialPropagatorParabolicMf_cff
import TrackingTools.MaterialEffects.MaterialPropagator_cfi
mixedTripletStepPropagator = TrackingTools.MaterialEffects.MaterialPropagator_cfi.MaterialPropagator.clone(
#mixedTripletStepPropagator = TrackingTools.MaterialEffects.MaterialPropagatorParabolicMf_cff.MaterialPropagatorParabolicMF.clone(
    ComponentName = 'mixedTripletStepPropagator',
    ptMin = 0.1
    )
for e in [pp_on_XeXe_2017, pp_on_AA_2018]:
    e.toModify(mixedTripletStepPropagator, ptMin=0.4)
highBetaStar_2018.toModify(mixedTripletStepPropagator,ptMin = 0.05)

import TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi
mixedTripletStepPropagatorOpposite = TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi.OppositeMaterialPropagator.clone(
#mixedTripletStepPropagatorOpposite = TrackingTools.MaterialEffects.MaterialPropagatorParabolicMf_cff.OppositeMaterialPropagatorParabolicMF.clone(
    ComponentName = 'mixedTripletStepPropagatorOpposite',
    ptMin = 0.1
    )
for e in [pp_on_XeXe_2017, pp_on_AA_2018]:
    e.toModify(mixedTripletStepPropagatorOpposite, ptMin=0.4)
highBetaStar_2018.toModify(mixedTripletStepPropagatorOpposite,ptMin = 0.05)

import RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi
mixedTripletStepChi2Est = RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi.Chi2ChargeMeasurementEstimator.clone(
    ComponentName = cms.string('mixedTripletStepChi2Est'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(16.0),
    clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight'))
)
trackingLowPU.toModify(mixedTripletStepChi2Est,
    clusterChargeCut = dict(refToPSet_ = 'SiStripClusterChargeCutTiny')
)

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
mixedTripletStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    MeasurementTrackerName = '',
    trajectoryFilter = cms.PSet(refToPSet_ = cms.string('mixedTripletStepTrajectoryFilter')),
    propagatorAlong = cms.string('mixedTripletStepPropagator'),
    propagatorOpposite = cms.string('mixedTripletStepPropagatorOpposite'),
    maxCand = 2,
    estimator = cms.string('mixedTripletStepChi2Est'),
    maxDPhiForLooperReconstruction = cms.double(2.0),
    maxPtForLooperReconstruction = cms.double(0.7) 
    )

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
mixedTripletStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('mixedTripletStepSeeds'),
    clustersToSkip = cms.InputTag('mixedTripletStepClusters'),
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = cms.int32(50),
    #onlyPixelHitsForSeedCleaner = cms.bool(True),

    TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string('mixedTripletStepTrajectoryBuilder')),
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True,
    TrajectoryCleaner = 'mixedTripletStepTrajectoryCleanerBySharedHits'
)
import FastSimulation.Tracking.TrackCandidateProducer_cfi
fastSim.toReplaceWith(mixedTripletStepTrackCandidates,
                      FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
        src = cms.InputTag("mixedTripletStepSeeds"),
        MinNumberOfCrossedLayers = 3,
        hitMasks = cms.InputTag("mixedTripletStepMasks"),
        )
)


from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi import trajectoryCleanerBySharedHits
mixedTripletStepTrajectoryCleanerBySharedHits = trajectoryCleanerBySharedHits.clone(
        ComponentName = cms.string('mixedTripletStepTrajectoryCleanerBySharedHits'),
            fractionShared = cms.double(0.11),
            allowSharedFirstHit = cms.bool(True)
            )
trackingLowPU.toModify(mixedTripletStepTrajectoryCleanerBySharedHits, fractionShared = 0.19)


# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
mixedTripletStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    AlgorithmName = cms.string('mixedTripletStep'),
    src = 'mixedTripletStepTrackCandidates',
    Fitter = cms.string('FlexibleKFFittingSmoother')
)
fastSim.toModify(mixedTripletStepTracks, TTRHBuilder = 'WithoutRefit')

# TRACK SELECTION AND QUALITY FLAG SETTING.
from RecoTracker.FinalTrackSelectors.TrackMVAClassifierPrompt_cfi import *
from RecoTracker.FinalTrackSelectors.TrackMVAClassifierDetached_cfi import *
mixedTripletStepClassifier1 = TrackMVAClassifierDetached.clone()
mixedTripletStepClassifier1.src = 'mixedTripletStepTracks'
mixedTripletStepClassifier1.mva.GBRForestLabel = 'MVASelectorIter4_13TeV'
mixedTripletStepClassifier1.qualityCuts = [-0.5,0.0,0.5]
fastSim.toModify(mixedTripletStepClassifier1, vertices = "firstStepPrimaryVerticesBeforeMixing")

mixedTripletStepClassifier2 = TrackMVAClassifierPrompt.clone()
mixedTripletStepClassifier2.src = 'mixedTripletStepTracks'
mixedTripletStepClassifier2.mva.GBRForestLabel = 'MVASelectorIter0_13TeV'
mixedTripletStepClassifier2.qualityCuts = [-0.2,-0.2,-0.2]
fastSim.toModify(mixedTripletStepClassifier2,vertices = "firstStepPrimaryVerticesBeforeMixing")

from RecoTracker.FinalTrackSelectors.ClassifierMerger_cfi import *
mixedTripletStep = ClassifierMerger.clone()
mixedTripletStep.inputClassifiers=['mixedTripletStepClassifier1','mixedTripletStepClassifier2']

trackingPhase1.toReplaceWith(mixedTripletStep, mixedTripletStepClassifier1.clone(
	mva = dict(GBRForestLabel = 'MVASelectorMixedTripletStep_Phase1'),
	qualityCuts = [-0.5,0.0,0.5]
))

from RecoTracker.FinalTrackSelectors.TrackLwtnnClassifier_cfi import *
from RecoTracker.FinalTrackSelectors.trackSelectionLwtnn_cfi import *
trackdnn.toReplaceWith(mixedTripletStep, TrackLwtnnClassifier.clone(
     src = 'mixedTripletStepTracks',
     qualityCuts = [-0.8, -0.35, 0.1]
))
(trackdnn & fastSim).toModify(mixedTripletStep,vertices = "firstStepPrimaryVerticesBeforeMixing")

highBetaStar_2018.toModify(mixedTripletStep,qualityCuts = [-0.7,0.0,0.5])
pp_on_AA_2018.toModify(mixedTripletStep, qualityCuts = [-0.5,0.0,0.9])

# For LowPU
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
mixedTripletStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src = 'mixedTripletStepTracks',
    useAnyMVA = cms.bool(False),
    GBRForestLabel = cms.string('MVASelectorIter4'),
    trackSelectors = [
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'mixedTripletStepVtxLoose',
            chi2n_par = 1.2,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 2,
            d0_par1 = ( 1.2, 3.0 ),
            dz_par1 = ( 1.2, 3.0 ),
            d0_par2 = ( 1.3, 3.0 ),
            dz_par2 = ( 1.3, 3.0 )
        ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'mixedTripletStepTrkLoose',
            chi2n_par = 0.6,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 4,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 3,
            d0_par1 = ( 1.2, 4.0 ),
            dz_par1 = ( 1.2, 4.0 ),
            d0_par2 = ( 1.2, 4.0 ),
            dz_par2 = ( 1.2, 4.0 )
        ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'mixedTripletStepVtxTight',
            preFilterName = 'mixedTripletStepVtxLoose',
            chi2n_par = 0.6,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 3,
            d0_par1 = ( 1.1, 3.0 ),
            dz_par1 = ( 1.1, 3.0 ),
            d0_par2 = ( 1.2, 3.0 ),
            dz_par2 = ( 1.2, 3.0 )
        ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'mixedTripletStepTrkTight',
            preFilterName = 'mixedTripletStepTrkLoose',
            chi2n_par = 0.4,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 5,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 4,
            d0_par1 = ( 1.1, 4.0 ),
            dz_par1 = ( 1.1, 4.0 ),
            d0_par2 = ( 1.1, 4.0 ),
            dz_par2 = ( 1.1, 4.0 )
        ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'mixedTripletStepVtx',
            preFilterName = 'mixedTripletStepVtxTight',
            chi2n_par = 0.4,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 3,
            d0_par1 = ( 1.1, 3.0 ),
            dz_par1 = ( 1.1, 3.0 ),
            d0_par2 = ( 1.2, 3.0 ),
            dz_par2 = ( 1.2, 3.0 )
        ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'mixedTripletStepTrk',
            preFilterName = 'mixedTripletStepTrkTight',
            chi2n_par = 0.3,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 5,
            maxNumberLostLayers = 0,
            minNumber3DLayers = 4,
            d0_par1 = ( 0.9, 4.0 ),
            dz_par1 = ( 0.9, 4.0 ),
            d0_par2 = ( 0.9, 4.0 ),
            dz_par2 = ( 0.9, 4.0 )
        )
    ] #end of vpset
) #end of clone


from RecoTracker.FinalTrackSelectors.trackAlgoPriorityOrder_cfi import trackAlgoPriorityOrder
import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
_trackListMergerBase = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = ['mixedTripletStepTracks',
                      'mixedTripletStepTracks'],
    hasSelector = [1,1],
    selectedTrackQuals = [cms.InputTag("mixedTripletStepSelector","mixedTripletStepVtx"),
                          cms.InputTag("mixedTripletStepSelector","mixedTripletStepTrk")],
    setsToMerge = [cms.PSet( tLists=cms.vint32(0,1), pQual=cms.bool(True) )],
    writeOnlyTrkQuals = True
)
trackingLowPU.toReplaceWith(mixedTripletStep, _trackListMergerBase)



MixedTripletStepTask = cms.Task(chargeCut2069Clusters,mixedTripletStepClusters,
                                mixedTripletStepSeedLayersA,
                                mixedTripletStepTrackingRegionsA,
                                mixedTripletStepHitDoubletsA,
                                mixedTripletStepHitTripletsA,
                                mixedTripletStepSeedsA,
                                mixedTripletStepSeedLayersB,
                                mixedTripletStepTrackingRegionsB,
                                mixedTripletStepHitDoubletsB,
                                mixedTripletStepHitTripletsB,
                                mixedTripletStepSeedsB,
                                mixedTripletStepSeeds,
                                mixedTripletStepTrackCandidates,
                                mixedTripletStepTracks,
                                mixedTripletStepClassifier1,mixedTripletStepClassifier2,
                                mixedTripletStep)
MixedTripletStep = cms.Sequence(MixedTripletStepTask)
_MixedTripletStepTask_LowPU = MixedTripletStepTask.copyAndExclude([chargeCut2069Clusters, mixedTripletStepClassifier1])
_MixedTripletStepTask_LowPU.replace(mixedTripletStepClassifier2, mixedTripletStepSelector)
trackingLowPU.toReplaceWith(MixedTripletStepTask, _MixedTripletStepTask_LowPU)

#fastsim
import FastSimulation.Tracking.FastTrackerRecHitMaskProducer_cfi
mixedTripletStepMasks = FastSimulation.Tracking.FastTrackerRecHitMaskProducer_cfi.maskProducerFromClusterRemover(mixedTripletStepClusters)
mixedTripletStepMasks.oldHitRemovalInfo = cms.InputTag("pixelPairStepMasks")

fastSim.toReplaceWith(MixedTripletStepTask,
                      cms.Task(mixedTripletStepMasks
                                   ,mixedTripletStepTrackingRegionsA
                                   ,mixedTripletStepSeedsA
                                   ,mixedTripletStepTrackingRegionsB
                                   ,mixedTripletStepSeedsB
                                   ,mixedTripletStepSeeds
                                   ,mixedTripletStepTrackCandidates
                                   ,mixedTripletStepTracks
                                   ,mixedTripletStepClassifier1,mixedTripletStepClassifier2
                                   ,mixedTripletStep                                 
                                   )
)
