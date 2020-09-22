import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_tracker_apv_vfp30_2016_cff import tracker_apv_vfp30_2016 as _tracker_apv_vfp30_2016
from Configuration.Eras.Modifier_fastSim_cff import fastSim

#for dnn classifier
from Configuration.ProcessModifiers.trackdnn_cff import trackdnn
from dnnQualityCuts import qualityCutDictionary

### STEP 0 ###

# hit building
from RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff import *
from RecoTracker.TransientTrackingRecHit.TTRHBuilders_cff import *

# SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi
import RecoTracker.TkSeedingLayers.PixelLayerQuadruplets_cfi
initialStepSeedLayers = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.PixelLayerTriplets.clone()
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
trackingPhase1.toModify(initialStepSeedLayers,
    layerList = RecoTracker.TkSeedingLayers.PixelLayerQuadruplets_cfi.PixelLayerQuadruplets.layerList.value()
)
trackingPhase2PU140.toModify(initialStepSeedLayers,
    layerList = RecoTracker.TkSeedingLayers.PixelLayerQuadruplets_cfi.PixelLayerQuadruplets.layerList.value()
)

# TrackingRegion
from RecoTracker.TkTrackingRegions.globalTrackingRegionFromBeamSpot_cfi import globalTrackingRegionFromBeamSpot as _globalTrackingRegionFromBeamSpot
initialStepTrackingRegions = _globalTrackingRegionFromBeamSpot.clone(RegionPSet = dict(
    ptMin        = 0.6,
    originRadius = 0.02,
    nSigmaZ      = 4.0
))
from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
trackingPhase1.toModify(initialStepTrackingRegions, RegionPSet = dict(ptMin = 0.5))
from Configuration.Eras.Modifier_highBetaStar_2018_cff import highBetaStar_2018
highBetaStar_2018.toModify(initialStepTrackingRegions,RegionPSet = dict(
     ptMin        = 0.05,
     originRadius = 0.2
))
trackingPhase2PU140.toModify(initialStepTrackingRegions, RegionPSet = dict(ptMin = 0.6,originRadius = 0.03))

# seeding
from RecoTracker.TkHitPairs.hitPairEDProducer_cfi import hitPairEDProducer as _hitPairEDProducer
initialStepHitDoublets = _hitPairEDProducer.clone(
    seedingLayers   = 'initialStepSeedLayers',
    trackingRegions = 'initialStepTrackingRegions',
    maxElement      = 50000000,
    produceIntermediateHitDoublets = True,
)
from RecoPixelVertexing.PixelTriplets.pixelTripletHLTEDProducer_cfi import pixelTripletHLTEDProducer as _pixelTripletHLTEDProducer
from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
import RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi
initialStepHitTriplets = _pixelTripletHLTEDProducer.clone(
    doublets              = 'initialStepHitDoublets',
    produceSeedingHitSets = True,
    SeedComparitorPSet = RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi.LowPtClusterShapeSeedComparitor.clone()
)
from RecoTracker.TkSeedGenerator.seedCreatorFromRegionConsecutiveHitsEDProducer_cff import seedCreatorFromRegionConsecutiveHitsEDProducer as _seedCreatorFromRegionConsecutiveHitsEDProducer
initialStepSeeds = _seedCreatorFromRegionConsecutiveHitsEDProducer.clone(
    seedingHitSets = 'initialStepHitTriplets',
)
from RecoPixelVertexing.PixelTriplets.caHitQuadrupletEDProducer_cfi import caHitQuadrupletEDProducer as _caHitQuadrupletEDProducer
_initialStepCAHitQuadruplets = _caHitQuadrupletEDProducer.clone(
    doublets = 'initialStepHitDoublets',
    extraHitRPhitolerance = initialStepHitTriplets.extraHitRPhitolerance,
    SeedComparitorPSet = initialStepHitTriplets.SeedComparitorPSet,
    maxChi2 = dict(
        pt1    = 0.7, pt2    = 2,
        value1 = 200, value2 = 50,
    ),
    useBendingCorrection = True,
    fitFastCircle        = True,
    fitFastCircleChi2Cut = True,
    CAThetaCut           = 0.0012,
    CAPhiCut             = 0.2,
)
highBetaStar_2018.toModify(_initialStepCAHitQuadruplets,
    CAThetaCut = 0.0024,
    CAPhiCut   = 0.4
)
initialStepHitQuadruplets = _initialStepCAHitQuadruplets.clone()

trackingPhase1.toModify(initialStepHitDoublets, layerPairs = [0,1,2]) # layer pairs (0,1), (1,2), (2,3)

trackingPhase2PU140.toModify(initialStepHitDoublets, layerPairs = [0,1,2]) # layer pairs (0,1), (1,2), (2,3)
trackingPhase2PU140.toModify(initialStepHitQuadruplets,
    CAThetaCut = 0.0010,
    CAPhiCut   = 0.175,
)

from RecoTracker.TkSeedGenerator.seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer_cff import seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer as _seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer
_initialStepSeedsConsecutiveHitsTripletOnly = _seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer.clone(
    seedingHitSets     = 'initialStepHitTriplets',
    SeedComparitorPSet = dict(# FIXME: is this defined in any cfi that could be imported instead of copy-paste?
        ComponentName = 'PixelClusterShapeSeedComparitor',
        FilterAtHelixStage = cms.bool(False),
        FilterPixelHits = cms.bool(True),
        FilterStripHits = cms.bool(False),
        ClusterShapeHitFilterName = cms.string('ClusterShapeHitFilter'),
        ClusterShapeCacheSrc = cms.InputTag('siPixelClusterShapeCache')
    ),
)
trackingPhase1.toReplaceWith(initialStepSeeds, _initialStepSeedsConsecutiveHitsTripletOnly.clone(
        seedingHitSets = 'initialStepHitQuadruplets'
))
trackingPhase2PU140.toReplaceWith(initialStepSeeds, _initialStepSeedsConsecutiveHitsTripletOnly.clone(
        seedingHitSets = 'initialStepHitQuadruplets'
))
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
from FastSimulation.Tracking.SeedingMigration import _hitSetProducerToFactoryPSet
_fastSim_initialStepSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    trackingRegions = 'initialStepTrackingRegions',
    seedFinderSelector = dict( pixelTripletGeneratorFactory = _hitSetProducerToFactoryPSet(initialStepHitTriplets),
                               layerList = initialStepSeedLayers.layerList.value())
)
_fastSim_initialStepSeeds.seedFinderSelector.pixelTripletGeneratorFactory.SeedComparitorPSet.ComponentName = 'none'
#new for phase1
trackingPhase1.toModify(_fastSim_initialStepSeeds, seedFinderSelector = dict(
        pixelTripletGeneratorFactory = None,
        CAHitQuadrupletGeneratorFactory = _hitSetProducerToFactoryPSet(initialStepHitQuadruplets).clone(SeedComparitorPSet = dict(ComponentName = 'none')),
        #new parameters required for phase1 seeding
        BPix = dict(
            TTRHBuilder = 'WithoutRefit',
            HitProducer = 'TrackingRecHitProducer',
            ),
        FPix = dict(
            TTRHBuilder = 'WithoutRefit',
            HitProducer = 'TrackingRecHitProducer',
            ),
        layerPairs = initialStepHitDoublets.layerPairs.value()
        )
)

fastSim.toReplaceWith(initialStepSeeds,_fastSim_initialStepSeeds)


# building
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
_initialStepTrajectoryFilterBase = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    minimumNumberOfHits = 3,
    minPt               = 0.2,
)
initialStepTrajectoryFilterBase = _initialStepTrajectoryFilterBase.clone(
    maxCCCLostHits = 0,
    minGoodStripCharge = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutLoose'))
)
from Configuration.Eras.Modifier_tracker_apv_vfp30_2016_cff import tracker_apv_vfp30_2016
_tracker_apv_vfp30_2016.toModify(initialStepTrajectoryFilterBase, maxCCCLostHits = 2)

from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
for e in [pp_on_XeXe_2017, pp_on_AA_2018]:
    e.toModify(initialStepTrajectoryFilterBase, minPt=0.6)
highBetaStar_2018.toModify(initialStepTrajectoryFilterBase, minPt = 0.05)

initialStepTrajectoryFilterInOut = initialStepTrajectoryFilterBase.clone(
    minimumNumberOfHits = 4,
    seedExtension       = 1,
    strictSeedExtension = True, # don't allow inactive
    pixelSeedExtension  = True,
)
from Configuration.Eras.Modifier_trackingLowPU_cff import trackingLowPU
trackingLowPU.toReplaceWith(initialStepTrajectoryFilterBase, _initialStepTrajectoryFilterBase)
trackingPhase2PU140.toReplaceWith(initialStepTrajectoryFilterBase, _initialStepTrajectoryFilterBase)

import RecoPixelVertexing.PixelLowPtUtilities.StripSubClusterShapeTrajectoryFilter_cfi
initialStepTrajectoryFilterShape = RecoPixelVertexing.PixelLowPtUtilities.StripSubClusterShapeTrajectoryFilter_cfi.StripSubClusterShapeTrajectoryFilterTIX12.clone()
initialStepTrajectoryFilter = cms.PSet(
    ComponentType = cms.string('CompositeTrajectoryFilter'),
    filters = cms.VPSet(
        cms.PSet( refToPSet_ = cms.string('initialStepTrajectoryFilterBase')),
    #    cms.PSet( refToPSet_ = cms.string('initialStepTrajectoryFilterShape'))
    ),
)

trackingPhase2PU140.toReplaceWith(initialStepTrajectoryFilter, TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    minimumNumberOfHits = 3,
    minPt               = 0.2
))
import RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi
initialStepChi2Est = RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi.Chi2ChargeMeasurementEstimator.clone(
    ComponentName = 'initialStepChi2Est',
    nSigma        = 3.0,
    MaxChi2       = 30.0,
    clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutLoose')),
    pTChargeCutThreshold = 15.
)
_tracker_apv_vfp30_2016.toModify(initialStepChi2Est,
    clusterChargeCut = dict(refToPSet_ = 'SiStripClusterChargeCutTiny')
)
trackingPhase2PU140.toModify(initialStepChi2Est,
    clusterChargeCut = dict(refToPSet_ = 'SiStripClusterChargeCutNone'),
)


import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
initialStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    trajectoryFilter = cms.PSet(refToPSet_ = cms.string('initialStepTrajectoryFilter')),
    alwaysUseInvalidHits = True,
    maxCand = 3,
    estimator = 'initialStepChi2Est',
    maxDPhiForLooperReconstruction = cms.double(2.0),
    maxPtForLooperReconstruction = cms.double(0.7)
)
trackingLowPU.toModify(initialStepTrajectoryBuilder, maxCand = 5)
trackingPhase1.toModify(initialStepTrajectoryBuilder,
    minNrOfHitsForRebuild = 1,
    keepOriginalIfRebuildFails = True,
)
trackingPhase2PU140.toModify(initialStepTrajectoryBuilder,
    minNrOfHitsForRebuild = 1,
    keepOriginalIfRebuildFails = True,
)

import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
initialStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = 'initialStepSeeds',
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = cms.int32(50),
    onlyPixelHitsForSeedCleaner = cms.bool(True),
    TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string('initialStepTrajectoryBuilder')),
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True
)

from Configuration.ProcessModifiers.trackingMkFit_cff import trackingMkFit
import RecoTracker.MkFit.mkFitInputConverter_cfi as mkFitInputConverter_cfi
import RecoTracker.MkFit.mkFitProducer_cfi as mkFitProducer_cfi
import RecoTracker.MkFit.mkFitOutputConverter_cfi as mkFitOutputConverter_cfi
initialStepTrackCandidatesMkFitInput = mkFitInputConverter_cfi.mkFitInputConverter.clone(
    seeds = 'initialStepSeeds',
)
initialStepTrackCandidatesMkFit = mkFitProducer_cfi.mkFitProducer.clone(
    hitsSeeds = 'initialStepTrackCandidatesMkFitInput',
)
trackingMkFit.toReplaceWith(initialStepTrackCandidates, mkFitOutputConverter_cfi.mkFitOutputConverter.clone(
    seeds = 'initialStepSeeds',
    hitsSeeds = 'initialStepTrackCandidatesMkFitInput',
    tracks = 'initialStepTrackCandidatesMkFit',
))

import FastSimulation.Tracking.TrackCandidateProducer_cfi
fastSim.toReplaceWith(initialStepTrackCandidates,
                      FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
        src = 'initialStepSeeds',
        MinNumberOfCrossedLayers = 3
))


# fitting
import RecoTracker.TrackProducer.TrackProducer_cfi
initialStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src           = 'initialStepTrackCandidates',
    AlgorithmName = 'initialStep',
    Fitter        = 'FlexibleKFFittingSmoother'
)
fastSim.toModify(initialStepTracks, TTRHBuilder = 'WithoutRefit')

#vertices
from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi import offlinePrimaryVertices as _offlinePrimaryVertices
firstStepPrimaryVerticesUnsorted = _offlinePrimaryVertices.clone(
    TrackLabel = 'initialStepTracks',
    vertexCollections = [_offlinePrimaryVertices.vertexCollections[0].clone()]
)
from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
(pp_on_XeXe_2017 | pp_on_AA_2018).toModify(firstStepPrimaryVerticesUnsorted, TkFilterParameters = dict(trackQuality = 'any'))

# we need a replacment for the firstStepPrimaryVerticesUnsorted
# that includes tracker information of signal and pile up
# after mixing there is no such thing as initialStepTracks,
# so we replace the input collection for firstStepPrimaryVerticesUnsorted with generalTracks
firstStepPrimaryVerticesBeforeMixing =  firstStepPrimaryVerticesUnsorted.clone()
fastSim.toModify(firstStepPrimaryVerticesUnsorted, TrackLabel = 'generalTracks')


from RecoJets.JetProducers.TracksForJets_cff import trackRefsForJets
initialStepTrackRefsForJets = trackRefsForJets.clone(
    src = 'initialStepTracks'
)
fastSim.toModify(initialStepTrackRefsForJets, src = 'generalTracks')
from RecoJets.JetProducers.caloJetsForTrk_cff import *
from CommonTools.RecoAlgos.sortedPrimaryVertices_cfi import sortedPrimaryVertices as _sortedPrimaryVertices
firstStepPrimaryVertices = _sortedPrimaryVertices.clone(
    vertices  = 'firstStepPrimaryVerticesUnsorted',
    particles = 'initialStepTrackRefsForJets',
)


# Final selection
from RecoTracker.FinalTrackSelectors.TrackMVAClassifierPrompt_cfi import *
from RecoTracker.FinalTrackSelectors.TrackMVAClassifierDetached_cfi import *

initialStepClassifier1 = TrackMVAClassifierPrompt.clone(
    src         = 'initialStepTracks',
    mva         = dict(GBRForestLabel = 'MVASelectorIter0_13TeV'),
    qualityCuts = [-0.9,-0.8,-0.7]
)
fastSim.toModify(initialStepClassifier1,vertices = 'firstStepPrimaryVerticesBeforeMixing')

from RecoTracker.IterativeTracking.DetachedTripletStep_cff import detachedTripletStepClassifier1
from RecoTracker.IterativeTracking.LowPtTripletStep_cff import lowPtTripletStep
initialStepClassifier2 = detachedTripletStepClassifier1.clone(
    src = 'initialStepTracks'
)
fastSim.toModify(initialStepClassifier2,vertices = 'firstStepPrimaryVerticesBeforeMixing')
initialStepClassifier3 = lowPtTripletStep.clone(
    src = 'initialStepTracks'
)
fastSim.toModify(initialStepClassifier3,vertices = 'firstStepPrimaryVerticesBeforeMixing')

from RecoTracker.FinalTrackSelectors.ClassifierMerger_cfi import *
initialStep = ClassifierMerger.clone(
    inputClassifiers=['initialStepClassifier1','initialStepClassifier2','initialStepClassifier3']
)
trackingPhase1.toReplaceWith(initialStep, initialStepClassifier1.clone(
     mva         = dict(GBRForestLabel = 'MVASelectorInitialStep_Phase1'),
     qualityCuts = [-0.95,-0.85,-0.75]
))

from RecoTracker.FinalTrackSelectors.TrackTfClassifier_cfi import *
from RecoTracker.FinalTrackSelectors.trackSelectionTf_cfi import *
trackdnn.toReplaceWith(initialStep, TrackTfClassifier.clone(
        src = 'initialStepTracks',
        qualityCuts = qualityCutDictionary["InitialStep"]
))

(trackdnn & fastSim).toModify(initialStep,vertices = "firstStepPrimaryVerticesBeforeMixing")

pp_on_AA_2018.toModify(initialStep, 
        mva         = dict(GBRForestLabel = 'HIMVASelectorInitialStep_Phase1'),
        qualityCuts = [-0.9, -0.5, 0.2],
)

# For LowPU and Phase2PU140
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
initialStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src = 'initialStepTracks',
    useAnyMVA = cms.bool(False),
    GBRForestLabel = cms.string('MVASelectorIter0'),
    trackSelectors = [
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'initialStepLoose',
        ), #end of pset
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'initialStepTight',
            preFilterName = 'initialStepLoose',
        ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'QualityMasks',
            preFilterName = 'initialStepTight',
        ),
    ] #end of vpset
) #end of clone
trackingPhase2PU140.toModify(initialStepSelector,
    useAnyMVA = None,
    GBRForestLabel = None,
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'initialStepLoose',
            chi2n_par = 2.0,
            res_par = ( 0.003, 0.002 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 3,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.8, 4.0 ),
            dz_par1 = ( 0.9, 4.0 ),
            d0_par2 = ( 0.6, 4.0 ),
            dz_par2 = ( 0.8, 4.0 )
            ), #end of pset
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'initialStepTight',
            preFilterName = 'initialStepLoose',
            chi2n_par = 1.4,
            res_par = ( 0.003, 0.002 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 2,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.7, 4.0 ),
            dz_par1 = ( 0.8, 4.0 ),
            d0_par2 = ( 0.5, 4.0 ),
            dz_par2 = ( 0.7, 4.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'initialStep',
            preFilterName = 'initialStepTight',
            min_eta = -4.1,
            max_eta = 4.1,            
            chi2n_par = 1.2,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 2,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.6, 4.0 ),
            dz_par1 = ( 0.7, 4.0 ),
            d0_par2 = ( 0.45, 4.0 ),
            dz_par2 = ( 0.55, 4.0 )
            ),
        ), #end of vpset
) #end of clone



# Final sequence
InitialStepTask = cms.Task(initialStepSeedLayers,
                           initialStepTrackingRegions,
                           initialStepHitDoublets,
                           initialStepHitTriplets,
                           initialStepSeeds,
                           initialStepTrackCandidates,
                           initialStepTracks,
                           firstStepPrimaryVerticesUnsorted,
                           initialStepTrackRefsForJets,
                           firstStepPrimaryVertices,
                           initialStepClassifier1,initialStepClassifier2,initialStepClassifier3,
                           initialStep,caloJetsForTrkTask)
InitialStep = cms.Sequence(InitialStepTask)

_InitialStepTask_trackingMkFit = InitialStepTask.copy()
_InitialStepTask_trackingMkFit.add(initialStepTrackCandidatesMkFitInput, initialStepTrackCandidatesMkFit)
trackingMkFit.toReplaceWith(InitialStepTask, _InitialStepTask_trackingMkFit)

_InitialStepTask_LowPU = InitialStepTask.copyAndExclude([firstStepPrimaryVerticesUnsorted, initialStepTrackRefsForJets, caloJetsForTrkTask, firstStepPrimaryVertices, initialStepClassifier1, initialStepClassifier2, initialStepClassifier3])
_InitialStepTask_LowPU.replace(initialStep, initialStepSelector)
trackingLowPU.toReplaceWith(InitialStepTask, _InitialStepTask_LowPU)

_InitialStepTask_Phase1 = InitialStepTask.copyAndExclude([initialStepClassifier2, initialStepClassifier3])
_InitialStepTask_Phase1.replace(initialStepHitTriplets, initialStepHitQuadruplets)
trackingPhase1.toReplaceWith(InitialStepTask, _InitialStepTask_Phase1)

_InitialStepTask_trackingPhase2 = InitialStepTask.copyAndExclude([initialStepClassifier1, initialStepClassifier2, initialStepClassifier3])
_InitialStepTask_trackingPhase2.replace(initialStepHitTriplets, initialStepHitQuadruplets)
_InitialStepTask_trackingPhase2.replace(initialStep, initialStepSelector)
trackingPhase2PU140.toReplaceWith(InitialStepTask, _InitialStepTask_trackingPhase2)

from Configuration.Eras.Modifier_fastSim_cff import fastSim
_InitialStepTask_fastSim = cms.Task(initialStepTrackingRegions
                           ,initialStepSeeds
                           ,initialStepTrackCandidates
                           ,initialStepTracks
                           ,firstStepPrimaryVerticesBeforeMixing
                           ,initialStepClassifier1,initialStepClassifier2,initialStepClassifier3
                           ,initialStep
                           )
fastSim.toReplaceWith(InitialStepTask, _InitialStepTask_fastSim)
