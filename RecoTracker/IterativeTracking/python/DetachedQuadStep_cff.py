import FWCore.ParameterSet.Config as cms
import RecoTracker.IterativeTracking.iterativeTkConfig as _cfg
from Configuration.Eras.Modifier_fastSim_cff import fastSim

#for dnn classifier
from Configuration.ProcessModifiers.trackdnn_cff import trackdnn
from RecoTracker.IterativeTracking.dnnQualityCuts import qualityCutDictionary

###############################################
# Low pT and detached tracks from pixel quadruplets
###############################################

# REMOVE HITS ASSIGNED TO GOOD TRACKS FROM PREVIOUS ITERATIONS
detachedQuadStepClusters = _cfg.clusterRemoverForIter('DetachedQuadStep')
for _eraName, _postfix, _era in _cfg.nonDefaultEras():
    _era.toReplaceWith(detachedQuadStepClusters, _cfg.clusterRemoverForIter('DetachedQuadStep', _eraName, _postfix))

# SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelLayerQuadruplets_cfi
detachedQuadStepSeedLayers = RecoTracker.TkSeedingLayers.PixelLayerQuadruplets_cfi.PixelLayerQuadruplets.clone(
    BPix = dict(skipClusters = cms.InputTag('detachedQuadStepClusters')),
    FPix = dict(skipClusters = cms.InputTag('detachedQuadStepClusters'))
)

# TrackingRegion
from RecoTracker.TkTrackingRegions.globalTrackingRegionFromBeamSpotFixedZ_cfi import globalTrackingRegionFromBeamSpotFixedZ as _globalTrackingRegionFromBeamSpotFixedZ
detachedQuadStepTrackingRegions = _globalTrackingRegionFromBeamSpotFixedZ.clone(RegionPSet = dict(
    ptMin            = 0.3,
    originHalfLength = 15.0,
    originRadius     = 1.5
))
from RecoTracker.TkTrackingRegions.globalTrackingRegionFromBeamSpot_cfi import globalTrackingRegionFromBeamSpot as _globalTrackingRegionFromBeamSpot
from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
trackingPhase2PU140.toReplaceWith(detachedQuadStepTrackingRegions, _globalTrackingRegionFromBeamSpot.clone(RegionPSet = dict(
    ptMin        = 0.45,
    originRadius = 0.9,
    nSigmaZ      = 5.0
)))

from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
from RecoTracker.TkTrackingRegions.globalTrackingRegionWithVertices_cff import globalTrackingRegionWithVertices as _globalTrackingRegionWithVertices
(pp_on_XeXe_2017 | pp_on_AA).toReplaceWith(detachedQuadStepTrackingRegions, 
                _globalTrackingRegionWithVertices.clone(RegionPSet=dict(
                    fixedError   = 3.75,
                    ptMin        = 0.9,
                    originRadius = 1.5
                )
                                                                      )
)
from Configuration.Eras.Modifier_highBetaStar_2018_cff import highBetaStar_2018
highBetaStar_2018.toModify(detachedQuadStepTrackingRegions,RegionPSet = dict(ptMin = 0.05))


# seeding
from RecoTracker.TkHitPairs.hitPairEDProducer_cfi import hitPairEDProducer as _hitPairEDProducer
detachedQuadStepHitDoublets = _hitPairEDProducer.clone(
    seedingLayers   = 'detachedQuadStepSeedLayers',
    trackingRegions = 'detachedQuadStepTrackingRegions',
    layerPairs      = [0,1,2], # layer pairs (0,1), (1,2), (2,3),
    maxElement      = 50000000,
    produceIntermediateHitDoublets = True,
)
from RecoPixelVertexing.PixelTriplets.caHitQuadrupletEDProducer_cfi import caHitQuadrupletEDProducer as _caHitQuadrupletEDProducer
from RecoPixelVertexing.PixelTriplets.pixelTripletLargeTipEDProducer_cfi import pixelTripletLargeTipEDProducer as _pixelTripletLargeTipEDProducer
from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
detachedQuadStepHitQuadruplets = _caHitQuadrupletEDProducer.clone(
    doublets = 'detachedQuadStepHitDoublets',
    extraHitRPhitolerance = _pixelTripletLargeTipEDProducer.extraHitRPhitolerance,
    maxChi2 = dict(
        pt1    = 0.8, pt2    = 2,
        value1 = 500, value2 = 100,
    ),
    useBendingCorrection = True,
    fitFastCircle        = True,
    fitFastCircleChi2Cut = True,
    CAThetaCut           = 0.0011,
    CAPhiCut             = 0,
)
highBetaStar_2018.toModify(detachedQuadStepHitQuadruplets,CAThetaCut = 0.0022,CAPhiCut = 0.1)

from RecoTracker.TkSeedGenerator.seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer_cff import seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer as _seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer
detachedQuadStepSeeds = _seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer.clone(
    seedingHitSets     = 'detachedQuadStepHitQuadruplets',
    SeedComparitorPSet = dict(# FIXME: is this defined in any cfi that could be imported instead of copy-paste?
        ComponentName  = 'PixelClusterShapeSeedComparitor',
        FilterAtHelixStage = cms.bool(False),
        FilterPixelHits    = cms.bool(True),
        FilterStripHits    = cms.bool(False),
        ClusterShapeHitFilterName = cms.string('ClusterShapeHitFilter'),
        ClusterShapeCacheSrc      = cms.InputTag('siPixelClusterShapeCache')
    ),
)

#For FastSim phase1 tracking
import FastSimulation.Tracking.TrajectorySeedProducer_cfi
from FastSimulation.Tracking.SeedingMigration import _hitSetProducerToFactoryPSet
_fastSim_detachedQuadStepSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone(
    trackingRegions = 'detachedQuadStepTrackingRegions',
    hitMasks        = cms.InputTag('detachedQuadStepMasks'),
    seedFinderSelector = dict( CAHitQuadrupletGeneratorFactory = _hitSetProducerToFactoryPSet(detachedQuadStepHitQuadruplets).clone(
            SeedComparitorPSet = dict(ComponentName = 'none')),
                               layerList  = detachedQuadStepSeedLayers.layerList.value(),
                               #new parameters required for phase1 seeding
                               BPix       = dict(TTRHBuilder = 'WithoutRefit', HitProducer = 'TrackingRecHitProducer',),
                               FPix       = dict(TTRHBuilder = 'WithoutRefit', HitProducer = 'TrackingRecHitProducer',),
                               layerPairs = detachedQuadStepHitDoublets.layerPairs.value()
                               ))
fastSim.toReplaceWith(detachedQuadStepSeeds,_fastSim_detachedQuadStepSeeds)

# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff as _TrajectoryFilter_cff
_detachedQuadStepTrajectoryFilterBase = _TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    minimumNumberOfHits = 3,
    minPt               = 0.075,
)
detachedQuadStepTrajectoryFilterBase = _detachedQuadStepTrajectoryFilterBase.clone(
    maxCCCLostHits     = 0,
    minGoodStripCharge = dict(refToPSet_ = 'SiStripClusterChargeCutLoose')
)
trackingPhase2PU140.toReplaceWith(detachedQuadStepTrajectoryFilterBase,
    _detachedQuadStepTrajectoryFilterBase.clone(
        maxLostHitsFraction = 1./10.,
        constantValueForLostHitsFractionFilter = 0.301,
    )
)
detachedQuadStepTrajectoryFilter = _TrajectoryFilter_cff.CompositeTrajectoryFilter_block.clone(
    filters = [cms.PSet(refToPSet_ = cms.string('detachedQuadStepTrajectoryFilterBase'))]
)
trackingPhase2PU140.toModify(detachedQuadStepTrajectoryFilter,
    filters = detachedQuadStepTrajectoryFilter.filters.value()+[cms.PSet(refToPSet_ = cms.string('ClusterShapeTrajectoryFilter'))]
)

(pp_on_XeXe_2017 | pp_on_AA).toModify(detachedQuadStepTrajectoryFilterBase, minPt=0.9)

import RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi
detachedQuadStepChi2Est = RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi.Chi2ChargeMeasurementEstimator.clone(
    ComponentName    = 'detachedQuadStepChi2Est',
    nSigma           = 3.0,
    MaxChi2          = 9.0,
    clusterChargeCut = dict(refToPSet_ = 'SiStripClusterChargeCutTight'),
)
trackingPhase2PU140.toModify(detachedQuadStepChi2Est,
    MaxChi2          = 12.0,
    clusterChargeCut = dict(refToPSet_ = 'SiStripClusterChargeCutNone')
)

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
detachedQuadStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    MeasurementTrackerName = '',
    trajectoryFilter = dict(refToPSet_ = 'detachedQuadStepTrajectoryFilter'),
    maxCand = 3,
    alwaysUseInvalidHits = True,
    estimator = 'detachedQuadStepChi2Est',
    maxDPhiForLooperReconstruction = cms.double(2.0),
    maxPtForLooperReconstruction = cms.double(0.7)
)
trackingPhase2PU140.toModify(detachedQuadStepTrajectoryBuilder,
    maxCand              = 2,
    alwaysUseInvalidHits = False,
)

# MAKING OF TRACK CANDIDATES
from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi import trajectoryCleanerBySharedHits
detachedQuadStepTrajectoryCleanerBySharedHits = trajectoryCleanerBySharedHits.clone(
    ComponentName       = 'detachedQuadStepTrajectoryCleanerBySharedHits',
    fractionShared      = 0.13,
    allowSharedFirstHit = True
)

import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
detachedQuadStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = 'detachedQuadStepSeeds',
    clustersToSkip = cms.InputTag('detachedQuadStepClusters'),
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = cms.int32(50),
    onlyPixelHitsForSeedCleaner = cms.bool(True),
    TrajectoryBuilderPSet = dict(refToPSet_ = 'detachedQuadStepTrajectoryBuilder'),
    TrajectoryCleaner = 'detachedQuadStepTrajectoryCleanerBySharedHits',
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True
)
trackingPhase2PU140.toModify(detachedQuadStepTrackCandidates,
    clustersToSkip       = None,
    phase2clustersToSkip = cms.InputTag('detachedQuadStepClusters')
)

from Configuration.ProcessModifiers.trackingMkFitDetachedQuadStep_cff import trackingMkFitDetachedQuadStep
import RecoTracker.MkFit.mkFitSeedConverter_cfi as mkFitSeedConverter_cfi
import RecoTracker.MkFit.mkFitIterationConfigESProducer_cfi as mkFitIterationConfigESProducer_cfi
import RecoTracker.MkFit.mkFitProducer_cfi as mkFitProducer_cfi
import RecoTracker.MkFit.mkFitOutputConverter_cfi as mkFitOutputConverter_cfi
detachedQuadStepTrackCandidatesMkFitSeeds = mkFitSeedConverter_cfi.mkFitSeedConverter.clone(
    seeds = 'detachedQuadStepSeeds',
)
detachedQuadStepTrackCandidatesMkFitConfig = mkFitIterationConfigESProducer_cfi.mkFitIterationConfigESProducer.clone(
    config = 'RecoTracker/MkFit/data/mkfit-phase1-detachedQuadStep.json',
    ComponentName = 'detachedQuadStepTrackCandidatesMkFitConfig',
)
detachedQuadStepTrackCandidatesMkFit = mkFitProducer_cfi.mkFitProducer.clone(
    seeds = 'detachedQuadStepTrackCandidatesMkFitSeeds',
    config = ('', 'detachedQuadStepTrackCandidatesMkFitConfig'),
    clustersToSkip = 'detachedQuadStepClusters',
)
trackingMkFitDetachedQuadStep.toReplaceWith(detachedQuadStepTrackCandidates, mkFitOutputConverter_cfi.mkFitOutputConverter.clone(
    seeds = 'detachedQuadStepSeeds',
    mkFitSeeds = 'detachedQuadStepTrackCandidatesMkFitSeeds',
    tracks = 'detachedQuadStepTrackCandidatesMkFit',
))

#For FastSim phase1 tracking 
import FastSimulation.Tracking.TrackCandidateProducer_cfi
_fastSim_detachedQuadStepTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
    src                      = 'detachedQuadStepSeeds',
    MinNumberOfCrossedLayers = 4,
    hitMasks                 = cms.InputTag('detachedQuadStepMasks')
)
fastSim.toReplaceWith(detachedQuadStepTrackCandidates,_fastSim_detachedQuadStepTrackCandidates)

# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
detachedQuadStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    AlgorithmName = 'detachedQuadStep',
    src           = 'detachedQuadStepTrackCandidates',
    Fitter        = 'FlexibleKFFittingSmoother',
)
fastSim.toModify(detachedQuadStepTracks,TTRHBuilder = 'WithoutRefit')

# TRACK SELECTION AND QUALITY FLAG SETTING.
from RecoTracker.FinalTrackSelectors.TrackMVAClassifierDetached_cfi import *
detachedQuadStep = TrackMVAClassifierDetached.clone(
    mva         = dict(GBRForestLabel = 'MVASelectorDetachedQuadStep_Phase1'),
    src         = 'detachedQuadStepTracks',
    qualityCuts = [-0.5,0.0,0.5]
)

from RecoTracker.FinalTrackSelectors.TrackTfClassifier_cfi import *
from RecoTracker.FinalTrackSelectors.trackSelectionTf_cfi import *
trackdnn.toReplaceWith(detachedQuadStep, TrackTfClassifier.clone(
    src = 'detachedQuadStepTracks',
    qualityCuts = qualityCutDictionary['DetachedQuadStep']
))

highBetaStar_2018.toModify(detachedQuadStep,qualityCuts = [-0.7,0.0,0.5])
pp_on_AA.toModify(detachedQuadStep, 
        mva         = dict(GBRForestLabel = 'HIMVASelectorDetachedQuadStep_Phase1'),
        qualityCuts = [-0.2, 0.2, 0.5],
)

fastSim.toModify(detachedQuadStep,vertices = 'firstStepPrimaryVerticesBeforeMixing')

# For Phase2PU140
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
detachedQuadStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src = 'detachedQuadStepTracks',
    trackSelectors = [
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'detachedQuadStepVtxLoose',
            chi2n_par = 1.0,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            d0_par1 = ( 0.9, 3.0 ),
            dz_par1 = ( 0.9, 3.0 ),
            d0_par2 = ( 1.0, 3.0 ),
            dz_par2 = ( 1.0, 3.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'detachedQuadStepTrkLoose',
            chi2n_par = 0.6,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            d0_par1 = ( 1.3, 4.0 ),
            dz_par1 = ( 1.3, 4.0 ),
            d0_par2 = ( 1.3, 4.0 ),
            dz_par2 = ( 1.3, 4.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'detachedQuadStepVtxTight',
            preFilterName = 'detachedQuadStepVtxLoose',
            chi2n_par = 0.9,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.9, 3.0 ),
            dz_par1 = ( 0.9, 3.0 ),
            d0_par2 = ( 0.9, 3.0 ),
            dz_par2 = ( 0.9, 3.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'detachedQuadStepTrkTight',
            preFilterName = 'detachedQuadStepTrkLoose',
            chi2n_par = 0.5,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 4,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 3,
            d0_par1 = ( 1.1, 4.0 ),
            dz_par1 = ( 1.1, 4.0 ),
            d0_par2 = ( 1.1, 4.0 ),
            dz_par2 = ( 1.1, 4.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'detachedQuadStepVtx',
            preFilterName = 'detachedQuadStepVtxTight',
            min_eta = -4.0, # it is particularly effective against fake tracks
            max_eta = 4.0,
            chi2n_par = 0.9,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.8, 3.0 ),
            dz_par1 = ( 0.8, 3.0 ),
            d0_par2 = ( 0.8, 3.0 ),
            dz_par2 = ( 0.8, 3.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'detachedQuadStepTrk',
            preFilterName = 'detachedQuadStepTrkTight',
            min_eta = -4.0, # it is particularly effective against fake tracks
            max_eta = 4.0,
            chi2n_par = 0.5,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 4,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.9, 4.0 ),
            dz_par1 = ( 0.9, 4.0 ),
            d0_par2 = ( 0.9, 4.0 ),
            dz_par2 = ( 0.9, 4.0 )
            )
    ] #end of vpset
) #end of clone


from RecoTracker.FinalTrackSelectors.trackAlgoPriorityOrder_cfi import trackAlgoPriorityOrder
import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
trackingPhase2PU140.toReplaceWith(detachedQuadStep, RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers     =['detachedQuadStepTracks',
                         'detachedQuadStepTracks'],
    hasSelector        =[1,1],
    ShareFrac          = 0.09,
    indivShareFrac     =[0.09,0.09],
    selectedTrackQuals =['detachedQuadStepSelector:detachedQuadStepVtx',
                         'detachedQuadStepSelector:detachedQuadStepTrk'],
    setsToMerge = cms.VPSet(cms.PSet( tLists=cms.vint32(0,1), pQual=cms.bool(True) )),
    writeOnlyTrkQuals  =True
    )
)

DetachedQuadStepTask = cms.Task(detachedQuadStepClusters,
                                detachedQuadStepSeedLayers,
                                detachedQuadStepTrackingRegions,
                                detachedQuadStepHitDoublets,
                                detachedQuadStepHitQuadruplets,
                                detachedQuadStepSeeds,
                                detachedQuadStepTrackCandidates,
                                detachedQuadStepTracks,
                                detachedQuadStep)
DetachedQuadStep = cms.Sequence(DetachedQuadStepTask)

_DetachedQuadStepTask_trackingMkFit = DetachedQuadStepTask.copy()
_DetachedQuadStepTask_trackingMkFit.add(detachedQuadStepTrackCandidatesMkFitSeeds, detachedQuadStepTrackCandidatesMkFit, detachedQuadStepTrackCandidatesMkFitConfig)
trackingMkFitDetachedQuadStep.toReplaceWith(DetachedQuadStepTask, _DetachedQuadStepTask_trackingMkFit)

_DetachedQuadStepTask_Phase2PU140 = DetachedQuadStepTask.copy()
_DetachedQuadStepTask_Phase2PU140.replace(detachedQuadStep, cms.Task(detachedQuadStepSelector,detachedQuadStep))
trackingPhase2PU140.toReplaceWith(DetachedQuadStepTask, _DetachedQuadStepTask_Phase2PU140)

#fastsim
from FastSimulation.Tracking.FastTrackerRecHitMaskProducer_cfi import maskProducerFromClusterRemover
detachedQuadStepMasks = maskProducerFromClusterRemover(detachedQuadStepClusters)
fastSim.toReplaceWith(DetachedQuadStepTask,
                      cms.Task(detachedQuadStepMasks
                               ,detachedQuadStepTrackingRegions
                               ,detachedQuadStepSeeds
                               ,detachedQuadStepTrackCandidates
                               ,detachedQuadStepTracks
                               ,detachedQuadStep
                               ) )
