import FWCore.ParameterSet.Config as cms

#for dnn classifier
from Configuration.ProcessModifiers.trackdnn_cff import trackdnn
from RecoTracker.IterativeTracking.dnnQualityCuts import qualityCutDictionary

# This step runs over all clusters

# run only if there are high pT jets
jetsForCoreTracking = cms.EDFilter('CandPtrSelector', src = cms.InputTag('ak4CaloJetsForTrk'), cut = cms.string('pt > 100 && abs(eta) < 2.5'))

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
    TIB = cms.PSet(
        matchedRecHits = cms.InputTag('siStripMatchedRecHits','matchedRecHit'),
        TTRHBuilder = cms.string('WithTrackAngle'), clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutNone'))
    ),
    BPix = cms.PSet(
        useErrorsFromParam = cms.bool(True),
        hitErrorRPhi = cms.double(0.0027),
        hitErrorRZ = cms.double(0.006),
        TTRHBuilder = cms.string('WithTrackAngle'),
        HitProducer = cms.string('siPixelRecHits'),
        #skipClusters = cms.InputTag('jetCoreRegionalStepClusters')
    ),
    FPix = cms.PSet(
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
jetCoreRegionalStepTrackingRegions = _tauRegionalPixelSeedTrackingRegions.clone(RegionPSet=dict(
    ptMin          = 10,
    deltaPhiRegion = 0.20,
    deltaEtaRegion = 0.20,
    JetSrc         = 'jetsForCoreTracking',
    vertexSrc      = 'firstStepGoodPrimaryVertices',
    howToUseMeasurementTracker = 'Never'
))

# Seeding
from RecoTracker.TkHitPairs.hitPairEDProducer_cfi import hitPairEDProducer as _hitPairEDProducer
jetCoreRegionalStepHitDoublets = _hitPairEDProducer.clone(
    seedingLayers         = 'jetCoreRegionalStepSeedLayers',
    trackingRegions       = 'jetCoreRegionalStepTrackingRegions',
    produceSeedingHitSets = True,
    maxElementTotal       = 12000000,
)
from RecoTracker.TkSeedGenerator.seedCreatorFromRegionConsecutiveHitsEDProducer_cff import seedCreatorFromRegionConsecutiveHitsEDProducer as _seedCreatorFromRegionConsecutiveHitsEDProducer
jetCoreRegionalStepSeeds = _seedCreatorFromRegionConsecutiveHitsEDProducer.clone(
    seedingHitSets = 'jetCoreRegionalStepHitDoublets',
    forceKinematicWithRegionDirection = True
)

# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
jetCoreRegionalStepTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    minimumNumberOfHits = 4,
    seedPairPenalty     = 0,
    minPt               = 0.1
)

from Configuration.ProcessModifiers.seedingDeepCore_cff import seedingDeepCore
seedingDeepCore.toModify(jetCoreRegionalStepTrajectoryFilter,
    minimumNumberOfHits = 2,
    maxConsecLostHits   = 2,
    maxLostHitsFraction = 1.1,
    minPt               = 0.9
    )

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
jetCoreRegionalStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    MeasurementTrackerName = '',
    trajectoryFilter = cms.PSet(refToPSet_ = cms.string('jetCoreRegionalStepTrajectoryFilter')),
    #clustersToSkip = cms.InputTag('jetCoreRegionalStepClusters'),
    maxCand = 50,
    estimator = 'jetCoreRegionalStepChi2Est',
    maxDPhiForLooperReconstruction = cms.double(2.0),
    maxPtForLooperReconstruction = cms.double(0.7)
    )
    
seedingDeepCore.toModify(jetCoreRegionalStepTrajectoryBuilder,
    maxPtForLooperReconstruction = 0.,
    keepOriginalIfRebuildFails = True,
    lockHits = False,
    requireSeedHitsInRebuild = False,
)

#customized cleaner for DeepCore
from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi import trajectoryCleanerBySharedHits
jetCoreRegionalStepDeepCoreTrajectoryCleaner = trajectoryCleanerBySharedHits.clone(
    ComponentName = 'jetCoreRegionalStepDeepCoreTrajectoryCleaner',
    fractionShared = 0.45
)

import RecoTracker.TkSeedGenerator.deepCoreSeedGenerator_cfi
import Validation.RecoTrack.JetCoreMCtruthSeedGenerator_cfi
seedingDeepCore.toReplaceWith(jetCoreRegionalStepSeeds,
    RecoTracker.TkSeedGenerator.deepCoreSeedGenerator_cfi.deepCoreSeedGenerator.clone(#to run MCtruthSeedGenerator clone here from Validation.RecoTrack
       vertices="firstStepPrimaryVertices" 
    )
)

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
jetCoreRegionalStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src                    = 'jetCoreRegionalStepSeeds',
    maxSeedsBeforeCleaning = 10000,
    TrajectoryBuilderPSet  = cms.PSet( refToPSet_ = cms.string('jetCoreRegionalStepTrajectoryBuilder')),
    NavigationSchool       = 'SimpleNavigationSchool',
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    #numHitsForSeedCleaner = cms.int32(50),
    #onlyPixelHitsForSeedCleaner = cms.bool(True),
)
seedingDeepCore.toModify(jetCoreRegionalStepTrackCandidates,
    TrajectoryCleaner         = 'jetCoreRegionalStepDeepCoreTrajectoryCleaner',
    doSeedingRegionRebuilding = True,    
)


# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
jetCoreRegionalStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    AlgorithmName = 'jetCoreRegionalStep',
    src           = 'jetCoreRegionalStepTrackCandidates',
    Fitter        = 'FlexibleKFFittingSmoother'
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
from RecoTracker.IterativeTracking.InitialStep_cff import initialStepClassifier1
#from RecoTracker.IterativeTracking.DetachedTripletStep_cff import detachedTripletStepClassifier1

#jetCoreRegionalStep = initialStepClassifier1.clone()
#jetCoreRegionalStep.src='jetCoreRegionalStepTracks'
#jetCoreRegionalStep.qualityCuts = [-0.3,0.0,0.2]
#jetCoreRegionalStep.vertices = 'firstStepGoodPrimaryVertices'

#jetCoreRegionalStepClassifier1 = initialStepClassifier1.clone()
#jetCoreRegionalStepClassifier1.src = 'jetCoreRegionalStepTracks'
#jetCoreRegionalStepClassifier1.qualityCuts = [-0.2,0.0,0.4]
#jetCoreRegionalStepClassifier2 = detachedTripletStepClassifier1.clone()
#jetCoreRegionalStepClassifier2.src = 'jetCoreRegionalStepTracks'



#from RecoTracker.FinalTrackSelectors.ClassifierMerger_cfi import *
#jetCoreRegionalStep = ClassifierMerger.clone()
#jetCoreRegionalStep.inputClassifiers=['jetCoreRegionalStepClassifier1','jetCoreRegionalStepClassifier2']


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
from RecoTracker.FinalTrackSelectors.TrackMVAClassifierPrompt_cfi import *

trackingPhase1.toReplaceWith(jetCoreRegionalStep, TrackMVAClassifierPrompt.clone(
     mva = dict(GBRForestLabel = 'MVASelectorJetCoreRegionalStep_Phase1'),
     src = 'jetCoreRegionalStepTracks',
     qualityCuts = [-0.2,0.0,0.4]
))

from RecoTracker.FinalTrackSelectors.TrackTfClassifier_cfi import *
from RecoTracker.FinalTrackSelectors.trackSelectionTf_cfi import *
trackdnn.toReplaceWith(jetCoreRegionalStep, TrackTfClassifier.clone(
     src = 'jetCoreRegionalStepTracks',
     qualityCuts = qualityCutDictionary["JetCoreRegionalStep"],
))

fastSim.toModify(jetCoreRegionalStep,vertices = 'firstStepPrimaryVerticesBeforeMixing')

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
#                                   jetCoreRegionalStepClassifier1,jetCoreRegionalStepClassifier2,
                                   jetCoreRegionalStep)
JetCoreRegionalStep = cms.Sequence(JetCoreRegionalStepTask)
seedingDeepCore.toReplaceWith(JetCoreRegionalStep,JetCoreRegionalStep.copyAndExclude([jetCoreRegionalStepHitDoublets]))
fastSim.toReplaceWith(JetCoreRegionalStepTask, 
                      cms.Task(jetCoreRegionalStepTracks,
                                   jetCoreRegionalStep))
