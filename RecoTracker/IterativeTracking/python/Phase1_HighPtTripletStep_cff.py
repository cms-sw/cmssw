import FWCore.ParameterSet.Config as cms

### high-pT triplets ###

# NEW CLUSTERS (remove previously used clusters)
from RecoLocalTracker.SubCollectionProducers.trackClusterRemover_cfi import *
highPtTripletStepClusters = trackClusterRemover.clone(
    maxChi2                                  = cms.double(9.0),
    trajectories                             = cms.InputTag("initialStepTracks"),
    pixelClusters                            = cms.InputTag("siPixelClusters"),
    stripClusters                            = cms.InputTag("siStripClusters"),
    oldClusterRemovalInfo                    = cms.InputTag(""),
    trackClassifier                          = cms.InputTag('initialStep',"QualityMasks"),
    TrackQuality                             = cms.string('highPurity'),
    minNumberOfLayersWithMeasBeforeFiltering = cms.int32(0),
)

# SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi
highPtTripletStepSeedLayers = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.PixelLayerTriplets.clone(
    layerList = [
        'BPix1+BPix2+BPix3',
        'BPix2+BPix3+BPix4',
        'BPix1+BPix3+BPix4',
        'BPix1+BPix2+BPix4',
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
)
highPtTripletStepSeedLayers.BPix.skipClusters = cms.InputTag('highPtTripletStepClusters')
highPtTripletStepSeedLayers.FPix.skipClusters = cms.InputTag('highPtTripletStepClusters')

# seeding
from RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff import *
from RecoTracker.TkTrackingRegions.GlobalTrackingRegionFromBeamSpot_cfi import RegionPsetFomBeamSpotBlock
highPtTripletStepSeeds = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone(
    RegionFactoryPSet = RegionPsetFomBeamSpotBlock.clone(
        ComponentName = cms.string('GlobalRegionProducerFromBeamSpot'),
        RegionPSet = RegionPsetFomBeamSpotBlock.RegionPSet.clone(
            ptMin = 0.6,
            originRadius = 0.02,
            nSigmaZ = 4.0
        )
    )
)
highPtTripletStepSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'highPtTripletStepSeedLayers'

from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
import RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi
highPtTripletStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.SeedComparitorPSet = RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi.LowPtClusterShapeSeedComparitor
highPtTripletStepSeeds.ClusterCheckPSet.doClusterCheck = cms.bool(False) # FIXME: cluster check condition to be updated

# building
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
highPtTripletStepTrajectoryFilterBase = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    minimumNumberOfHits = 3,
    minPt = 0.2,
    maxCCCLostHits = 2,
    minGoodStripCharge = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutLoose'))
    )
import RecoPixelVertexing.PixelLowPtUtilities.StripSubClusterShapeTrajectoryFilter_cfi
highPtTripletStepTrajectoryFilter = cms.PSet(
    ComponentType = cms.string('CompositeTrajectoryFilter'),
    filters = cms.VPSet(
        cms.PSet( refToPSet_ = cms.string('highPtTripletStepTrajectoryFilterBase')),
    ),
)

import RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi
highPtTripletStepChi2Est = RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi.Chi2ChargeMeasurementEstimator.clone(
    ComponentName = cms.string('highPtTripletStepChi2Est'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(30.0),
    clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTiny')),
    pTChargeCutThreshold = cms.double(15.)
)

import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
highPtTripletStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    trajectoryFilter = cms.PSet(refToPSet_ = cms.string('highPtTripletStepTrajectoryFilter')),
    alwaysUseInvalidHits = True,
    maxCand = 3,
    estimator = cms.string('highPtTripletStepChi2Est'),
    maxDPhiForLooperReconstruction = cms.double(2.0),
    maxPtForLooperReconstruction = cms.double(0.7)
    )

import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
highPtTripletStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('highPtTripletStepSeeds'),
    clustersToSkip = cms.InputTag('highPtTripletStepClusters'),
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = cms.int32(50),
    onlyPixelHitsForSeedCleaner = cms.bool(True),
    TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string('highPtTripletStepTrajectoryBuilder')),
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True
    )

# fitting
import RecoTracker.TrackProducer.TrackProducer_cfi
highPtTripletStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'highPtTripletStepTrackCandidates',
    AlgorithmName = cms.string('highPtTripletStep'),
    Fitter = cms.string('FlexibleKFFittingSmoother'),
    TTRHBuilder=cms.string('WithTrackAngle') # FIXME: to be updated once we get the templates to GT
    )


# Final selection
# MVA selection to be enabled after re-training, for time being we go with cut-based selector
#from RecoTracker.FinalTrackSelectors.TrackMVAClassifierPrompt_cfi import *
#from RecoTracker.FinalTrackSelectors.TrackMVAClassifierDetached_cfi import *
#
#highPtTripletStepClassifier1 = TrackMVAClassifierPrompt.clone()
#highPtTripletStepClassifier1.src = 'highPtTripletStepTracks'
#highPtTripletStepClassifier1.GBRForestLabel = 'MVASelectorIter0_13TeV'
#highPtTripletStepClassifier1.qualityCuts = [-0.9,-0.8,-0.7]
#
#from RecoTracker.IterativeTracking.Phase1_DetachedTripletStep_cff import detachedTripletStepClassifier1
#from RecoTracker.IterativeTracking.Phase1_LowPtTripletStep_cff import lowPtTripletStep
#highPtTripletStepClassifier2 = detachedTripletStepClassifier1.clone()
#highPtTripletStepClassifier2.src = 'highPtTripletStepTracks'
#highPtTripletStepClassifier3 = lowPtTripletStep.clone()
#highPtTripletStepClassifier3.src = 'highPtTripletStepTracks'
#
#
#from RecoTracker.FinalTrackSelectors.ClassifierMerger_cfi import *
#highPtTripletStep = ClassifierMerger.clone()
#highPtTripletStep.inputClassifiers=['highPtTripletStepClassifier1','highPtTripletStepClassifier2','highPtTripletStepClassifier3']
#highPtTripletStep.inputClassifiers=['highPtTripletStepClassifier1','highPtTripletStepClassifier2','highPtTripletStepClassifier3']

from RecoTracker.FinalTrackSelectors.TrackCutClassifier_cfi import TrackCutClassifier
highPtTripletStep = TrackCutClassifier.clone(
    src = "highPtTripletStepTracks",
    vertices = "firstStepPrimaryVertices",
)
highPtTripletStep.mva.minPixelHits = [1,1,1]
highPtTripletStep.mva.maxChi2 = [9999.,9999.,9999.]
highPtTripletStep.mva.maxChi2n = [2.0,1.0,0.7]
highPtTripletStep.mva.minLayers = [3,3,3]
highPtTripletStep.mva.min3DLayers = [3,3,3]
highPtTripletStep.mva.maxLostLayers = [3,2,2]
highPtTripletStep.mva.dr_par.dr_par1 = [0.7,0.6,0.5]
highPtTripletStep.mva.dz_par.dz_par1 = [0.8,0.7,0.7]
highPtTripletStep.mva.dr_par.dr_par2 = [0.4,0.35,0.25]
highPtTripletStep.mva.dz_par.dz_par2 = [0.6,0.5,0.4]
highPtTripletStep.mva.dr_par.d0err_par = [0.002,0.002,0.001]


# Final sequence
HighPtTripletStep = cms.Sequence(highPtTripletStepClusters*
                                 highPtTripletStepSeedLayers*
                                 highPtTripletStepSeeds*
                                 highPtTripletStepTrackCandidates*
                                 highPtTripletStepTracks*
#                                 highPtTripletStepClassifier1*highPtTripletStepClassifier2*highPtTripletStepClassifier3*
                                 highPtTripletStep)
