import FWCore.ParameterSet.Config as cms

# for dnn classifier
from Configuration.ProcessModifiers.trackdnn_cff import trackdnn
from RecoTracker.IterativeTracking.dnnQualityCuts import qualityCutDictionary

# for no-loopers
from Configuration.ProcessModifiers.trackingNoLoopers_cff import trackingNoLoopers

# This step runs over all clusters

# run only if there are high pT jets
jetsForCoreTracking = cms.EDFilter('CandPtrSelector', src = cms.InputTag('ak4CaloJetsForTrk'), cut = cms.string('pt > 100 && abs(eta) < 2.5'), filter = cms.bool(False))

jetsForCoreTrackingBarrel = jetsForCoreTracking.clone( cut = 'pt > 100 && abs(eta) < 2.5' )
jetsForCoreTrackingEndcap = jetsForCoreTracking.clone( cut = 'pt > 100 && abs(eta) > 1.4 && abs(eta) < 2.5' )

# care only at tracks from main PV
firstStepGoodPrimaryVertices = cms.EDFilter('PrimaryVertexObjectFilter',
     filterParams = cms.PSet(
     	     minNdof = cms.double(25.0),
             maxZ = cms.double(15.0),
             maxRho = cms.double(2.0)
     ),
     src=cms.InputTag('firstStepPrimaryVertices')
)

import RecoTracker.TkSeedingLayers.seedingLayersEDProducer_cfi as _mod

# SEEDING LAYERS
jetCoreRegionalStepSeedLayers = _mod.seedingLayersEDProducer.clone(
    layerList = ['BPix1+BPix2', 'BPix1+BPix3', 'BPix2+BPix3', 
                 'BPix1+FPix1_pos', 'BPix1+FPix1_neg', 
                 'BPix2+FPix1_pos', 'BPix2+FPix1_neg', 
                 'FPix1_pos+FPix2_pos', 'FPix1_neg+FPix2_neg',
                 #'BPix2+TIB1','BPix2+TIB2',
                 'BPix3+TIB1','BPix3+TIB2'],
    TIB = dict(
        matchedRecHits = cms.InputTag('siStripMatchedRecHits','matchedRecHit'),
        TTRHBuilder = cms.string('WithTrackAngle'), 
        clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutNone'))
    ),
    BPix = dict(
        useErrorsFromParam = cms.bool(True),
        hitErrorRPhi = cms.double(0.0027),
        hitErrorRZ = cms.double(0.006),
        TTRHBuilder = cms.string('WithTrackAngle'),
        HitProducer = cms.string('siPixelRecHits'),
        #skipClusters = cms.InputTag('jetCoreRegionalStepClusters')
    ),
    FPix = dict(
        useErrorsFromParam = cms.bool(True),
        hitErrorRPhi = cms.double(0.0051),
        hitErrorRZ = cms.double(0.0036),
        TTRHBuilder = cms.string('WithTrackAngle'),
        HitProducer = cms.string('siPixelRecHits'),
        #skipClusters = cms.InputTag('jetCoreRegionalStepClusters')
    )
)
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
_layerListForPhase1 = [
        'BPix1+BPix2', 'BPix1+BPix3', 'BPix1+BPix4',
        'BPix2+BPix3', 'BPix2+BPix4',
        'BPix3+BPix4',
        'BPix1+FPix1_pos', 'BPix1+FPix1_neg',
        'BPix2+FPix1_pos', 'BPix2+FPix1_neg',
        'FPix1_pos+FPix2_pos', 'FPix1_neg+FPix2_neg',
        'FPix1_pos+FPix3_pos', 'FPix1_neg+FPix3_neg',
        'FPix2_pos+FPix3_pos', 'FPix2_neg+FPix3_neg',
        #'BPix3+TIB1','BPix3+TIB2'
        'BPix4+TIB1','BPix4+TIB2'
    ]
trackingPhase1.toModify(jetCoreRegionalStepSeedLayers, layerList = _layerListForPhase1)

# TrackingRegion
from RecoTauTag.HLTProducers.tauRegionalPixelSeedTrackingRegions_cfi import tauRegionalPixelSeedTrackingRegions as _tauRegionalPixelSeedTrackingRegions
jetCoreRegionalStepTrackingRegions = _tauRegionalPixelSeedTrackingRegions.clone(
    RegionPSet=dict(
        ptMin          = 10,
        deltaPhiRegion = 0.20,
        deltaEtaRegion = 0.20,
        JetSrc         = 'jetsForCoreTracking',
        vertexSrc      = 'firstStepGoodPrimaryVertices',
        howToUseMeasurementTracker = 'Never')
)
jetCoreRegionalStepEndcapTrackingRegions = jetCoreRegionalStepTrackingRegions.clone(
    RegionPSet=dict(
        JetSrc = 'jetsForCoreTrackingEndcap')
)

# Seeding
from RecoTracker.TkHitPairs.hitPairEDProducer_cfi import hitPairEDProducer as _hitPairEDProducer
jetCoreRegionalStepHitDoublets = _hitPairEDProducer.clone(
    seedingLayers         = 'jetCoreRegionalStepSeedLayers',
    trackingRegions       = 'jetCoreRegionalStepTrackingRegions',
    produceSeedingHitSets = True,
    maxElementTotal       = 12000000,
)
jetCoreRegionalStepEndcapHitDoublets = jetCoreRegionalStepHitDoublets.clone(
    trackingRegions = 'jetCoreRegionalStepEndcapTrackingRegions',
)

from RecoTracker.TkSeedGenerator.seedCreatorFromRegionConsecutiveHitsEDProducer_cff import seedCreatorFromRegionConsecutiveHitsEDProducer as _seedCreatorFromRegionConsecutiveHitsEDProducer
jetCoreRegionalStepSeeds = _seedCreatorFromRegionConsecutiveHitsEDProducer.clone(
    seedingHitSets = 'jetCoreRegionalStepHitDoublets',
    forceKinematicWithRegionDirection = True
)
import RecoTracker.TkSeedGenerator.deepCoreSeedGenerator_cfi
jetCoreRegionalStepSeedsBarrel = RecoTracker.TkSeedGenerator.deepCoreSeedGenerator_cfi.deepCoreSeedGenerator.clone(#to run MCtruthSeedGenerator clone here from Validation.RecoTrack
    vertices = "firstStepPrimaryVertices",
    cores    = "jetsForCoreTrackingBarrel"
)

jetCoreRegionalStepSeedsEndcap = jetCoreRegionalStepSeeds.clone(
    seedingHitSets = 'jetCoreRegionalStepEndcapHitDoublets',
)

# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
jetCoreRegionalStepTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    minimumNumberOfHits = 4,
    seedPairPenalty     = 0,
    minPt               = 0.1
)
jetCoreRegionalStepBarrelTrajectoryFilter = jetCoreRegionalStepTrajectoryFilter.clone(
    minimumNumberOfHits = 2,
    maxConsecLostHits   = 2,
    maxLostHitsFraction = 1.1,
    seedPairPenalty     = 0,
    minPt               = 0.9 ## should it be slightly decrease ?
)
jetCoreRegionalStepEndcapTrajectoryFilter = jetCoreRegionalStepTrajectoryFilter.clone()


from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
(pp_on_XeXe_2017 | pp_on_AA).toModify(jetCoreRegionalStepTrajectoryFilter, minPt=5.0)

import TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi
jetCoreRegionalStepChi2Est = TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi.Chi2MeasurementEstimator.clone(
    ComponentName = 'jetCoreRegionalStepChi2Est',
    nSigma        = 3.0,
    MaxChi2       = 30.0
)

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
#need to also load the refToPSet_ used by GroupedCkfTrajectoryBuilder
CkfBaseTrajectoryFilter_block = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.CkfBaseTrajectoryFilter_block
jetCoreRegionalStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilderIterativeDefault.clone(
    trajectoryFilter = dict(refToPSet_ = 'jetCoreRegionalStepTrajectoryFilter'),
    maxCand = 50,
    estimator = 'jetCoreRegionalStepChi2Est',
    maxDPhiForLooperReconstruction = 2.0,
    maxPtForLooperReconstruction = 0.7,
)
trackingNoLoopers.toModify(jetCoreRegionalStepTrajectoryBuilder,
                           maxPtForLooperReconstruction = 0.0)    
jetCoreRegionalStepBarrelTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilderIterativeDefault.clone(
    trajectoryFilter = dict(refToPSet_ = 'jetCoreRegionalStepBarrelTrajectoryFilter'),
    maxCand = 50,
    estimator = 'jetCoreRegionalStepChi2Est',
    keepOriginalIfRebuildFails = True,
    lockHits = False,
    requireSeedHitsInRebuild = False,
)
trackingNoLoopers.toModify(jetCoreRegionalStepBarrelTrajectoryBuilder,
                           maxPtForLooperReconstruction = cms.double(0.0))    
jetCoreRegionalStepEndcapTrajectoryBuilder = jetCoreRegionalStepTrajectoryBuilder.clone(
    trajectoryFilter = cms.PSet(refToPSet_ = cms.string('jetCoreRegionalStepEndcapTrajectoryFilter')),
    #clustersToSkip = cms.InputTag('jetCoreRegionalStepClusters'),
)
trackingNoLoopers.toModify(jetCoreRegionalStepEndcapTrajectoryBuilder,
                           maxPtForLooperReconstruction = cms.double(0.0))
#customized cleaner for DeepCore
from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi import trajectoryCleanerBySharedHits
jetCoreRegionalStepDeepCoreTrajectoryCleaner = trajectoryCleanerBySharedHits.clone(
    ComponentName = 'jetCoreRegionalStepDeepCoreTrajectoryCleaner',
    fractionShared = 0.45
)

############## to run MCtruthSeedGenerator ####################
#import RecoTracker.TkSeedGenerator.deepCoreSeedGenerator_cfi
#import Validation.RecoTrack.JetCoreMCtruthSeedGenerator_cfi
#seedingDeepCore.toReplaceWith(jetCoreRegionalStepSeedsBarrel,
#    RecoTracker.TkSeedGenerator.deepCoreSeedGenerator_cfi.deepCoreSeedGenerator.clone(#to run MCtruthSeedGenerator clone here from Validation.RecoTrack
#       vertices="firstStepPrimaryVertices" 
#    )
#)


# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
jetCoreRegionalStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidatesIterativeDefault.clone(
    src                    = 'jetCoreRegionalStepSeeds',
    maxSeedsBeforeCleaning = 10000,
    TrajectoryBuilderPSet  = dict(refToPSet_ = 'jetCoreRegionalStepTrajectoryBuilder'),
    NavigationSchool       = 'SimpleNavigationSchool',
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    #numHitsForSeedCleaner = 50,
    #onlyPixelHitsForSeedCleaner = True,
)
jetCoreRegionalStepBarrelTrackCandidates = jetCoreRegionalStepTrackCandidates.clone(
    src                    = 'jetCoreRegionalStepSeedsBarrel',
    TrajectoryBuilderPSet  = cms.PSet( refToPSet_ = cms.string('jetCoreRegionalStepBarrelTrajectoryBuilder')),
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    #numHitsForSeedCleaner = cms.int32(50),
    #onlyPixelHitsForSeedCleaner = cms.bool(True),
    TrajectoryCleaner         = 'jetCoreRegionalStepDeepCoreTrajectoryCleaner',
    doSeedingRegionRebuilding = True,
)
jetCoreRegionalStepEndcapTrackCandidates = jetCoreRegionalStepTrackCandidates.clone(
    src                    = 'jetCoreRegionalStepSeedsEndcap',
    TrajectoryBuilderPSet  = cms.PSet( refToPSet_ = cms.string('jetCoreRegionalStepEndcapTrajectoryBuilder')),
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    #numHitsForSeedCleaner = cms.int32(50),
    #onlyPixelHitsForSeedCleaner = cms.bool(True),
)

# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducerIterativeDefault_cfi
jetCoreRegionalStepTracks = RecoTracker.TrackProducer.TrackProducerIterativeDefault_cfi.TrackProducerIterativeDefault.clone(
    AlgorithmName = 'jetCoreRegionalStep',
    src           = 'jetCoreRegionalStepTrackCandidates',
    Fitter        = 'FlexibleKFFittingSmoother'
)
jetCoreRegionalStepBarrelTracks = jetCoreRegionalStepTracks.clone(
    src           = 'jetCoreRegionalStepBarrelTrackCandidates',
)
jetCoreRegionalStepEndcapTracks = jetCoreRegionalStepTracks.clone(
    src           = 'jetCoreRegionalStepEndcapTrackCandidates',
)

from Configuration.Eras.Modifier_fastSim_cff import fastSim
import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
_fastSim_jetCoreRegionalStepTracks = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers     = [],
    hasSelector        = [],
    selectedTrackQuals = [],
    copyExtras         = True
)
fastSim.toReplaceWith(jetCoreRegionalStepTracks,_fastSim_jetCoreRegionalStepTracks)


# Final selection
from RecoTracker.FinalTrackSelectors.TrackCutClassifier_cff import *
jetCoreRegionalStep = TrackCutClassifier.clone(
    src = 'jetCoreRegionalStepTracks',
    mva = dict(
	minPixelHits  = [1,1,1],
        maxChi2       = [9999.,9999.,9999.],
        maxChi2n      = [1.6,1.0,0.7],
        minLayers     = [3,5,5],
        min3DLayers   = [1,2,3],
        maxLostLayers = [4,3,2],
        maxDz         = [0.5,0.35,0.2],
        maxDr         = [0.3,0.2,0.1]
    ),
    vertices = 'firstStepGoodPrimaryVertices'
)
jetCoreRegionalStepBarrel = jetCoreRegionalStep.clone(
    src = 'jetCoreRegionalStepBarrelTracks',
    mva = dict(
#	minPixelHits  = [1,1,1], # they could be easily increased to at least 2 or 3 !
        min3DLayers   = [1,2,2],
    ),
)

from RecoTracker.FinalTrackSelectors.TrackMVAClassifierPrompt_cfi import *
trackingPhase1.toReplaceWith(jetCoreRegionalStep, TrackMVAClassifierPrompt.clone(
     mva = dict(GBRForestLabel = 'MVASelectorJetCoreRegionalStep_Phase1'),
     src = 'jetCoreRegionalStepTracks',
     qualityCuts = [-0.2,0.0,0.4]
))

trackingPhase1.toReplaceWith(jetCoreRegionalStepBarrel, jetCoreRegionalStep.clone(
     src = 'jetCoreRegionalStepBarrelTracks',
))

pp_on_AA.toModify(jetCoreRegionalStep, qualityCuts = [-0.2, 0.0, 0.8])
pp_on_AA.toModify(jetCoreRegionalStepBarrel, qualityCuts = [-0.2, 0.0, 0.8])

from RecoTracker.FinalTrackSelectors.trackTfClassifier_cfi import *
from RecoTracker.FinalTrackSelectors.trackSelectionTf_cfi import *
from RecoTracker.FinalTrackSelectors.trackSelectionTf_CKF_cfi import *
trackdnn.toReplaceWith(jetCoreRegionalStep, trackTfClassifier.clone(
     src = 'jetCoreRegionalStepTracks',
     qualityCuts = qualityCutDictionary.JetCoreRegionalStep.value()
))
trackdnn.toReplaceWith(jetCoreRegionalStepBarrel, trackTfClassifier.clone(
     src = 'jetCoreRegionalStepBarrelTracks',
     qualityCuts = qualityCutDictionary.JetCoreRegionalStep.value()
))

fastSim.toModify(jetCoreRegionalStep,vertices = 'firstStepPrimaryVerticesBeforeMixing')


jetCoreRegionalStepEndcap = jetCoreRegionalStep.clone(
    src = 'jetCoreRegionalStepEndcapTracks',
)

# Final sequence
JetCoreRegionalStepTask = cms.Task(jetsForCoreTracking,                 
                                   firstStepGoodPrimaryVertices,
                                   #jetCoreRegionalStepClusters,
                                   jetCoreRegionalStepSeedLayers,
                                   jetCoreRegionalStepTrackingRegions,
                                   jetCoreRegionalStepHitDoublets,
                                   jetCoreRegionalStepSeeds,
                                   jetCoreRegionalStepTrackCandidates,
                                   jetCoreRegionalStepTracks,
                                   #jetCoreRegionalStepClassifier1,jetCoreRegionalStepClassifier2,
                                   jetCoreRegionalStep)
JetCoreRegionalStep = cms.Sequence(JetCoreRegionalStepTask)

JetCoreRegionalStepBarrelTask = cms.Task(jetsForCoreTrackingBarrel,
                                         firstStepGoodPrimaryVertices,
                                         #jetCoreRegionalStepClusters,
                                         jetCoreRegionalStepSeedLayers,
                                         jetCoreRegionalStepSeedsBarrel,
                                         jetCoreRegionalStepBarrelTrackCandidates,
                                         jetCoreRegionalStepBarrelTracks,
                                         jetCoreRegionalStepBarrel)

JetCoreRegionalStepEndcapTask = cms.Task(jetsForCoreTrackingEndcap,
                                         firstStepGoodPrimaryVertices,
                                         #jetCoreRegionalStepClusters,
                                         jetCoreRegionalStepSeedLayers,
                                         jetCoreRegionalStepEndcapTrackingRegions,
                                         jetCoreRegionalStepEndcapHitDoublets,
                                         jetCoreRegionalStepSeedsEndcap,
                                         jetCoreRegionalStepEndcapTrackCandidates,
                                         jetCoreRegionalStepEndcapTracks,
                                         jetCoreRegionalStepEndcap)


from Configuration.ProcessModifiers.seedingDeepCore_cff import seedingDeepCore

from RecoTracker.FinalTrackSelectors.TrackCollectionMerger_cfi import *
seedingDeepCore.toReplaceWith(jetCoreRegionalStepTracks, TrackCollectionMerger.clone(
    trackProducers   = ["jetCoreRegionalStepBarrelTracks",
                        "jetCoreRegionalStepEndcapTracks",],
    inputClassifiers = ["jetCoreRegionalStepBarrel",
                        "jetCoreRegionalStepEndcap",],
    foundHitBonus    = 100.0,
    lostHitPenalty   = 1.0
))

seedingDeepCore.toReplaceWith(jetCoreRegionalStep, jetCoreRegionalStepTracks.clone()) #(*)

seedingDeepCore.toReplaceWith(JetCoreRegionalStepTask, cms.Task(
    JetCoreRegionalStepBarrelTask,
    JetCoreRegionalStepEndcapTask,
    cms.Task(jetCoreRegionalStepTracks,jetCoreRegionalStep)
))

fastSim.toReplaceWith(JetCoreRegionalStepTask, 
                      cms.Task(jetCoreRegionalStepTracks,
                                   jetCoreRegionalStep))
