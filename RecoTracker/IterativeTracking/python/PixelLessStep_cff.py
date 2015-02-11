import FWCore.ParameterSet.Config as cms

##########################################################################
# Large impact parameter tracking using TIB/TID/TEC stereo layer seeding #
##########################################################################

from RecoLocalTracker.SubCollectionProducers.trackClusterRemover_cfi import *
pixelLessStepClusters = trackClusterRemover.clone(
    maxChi2                                  = cms.double(9.0),
    trajectories                             = cms.InputTag("mixedTripletStepTracks"),
    pixelClusters                            = cms.InputTag("siPixelClusters"),
    stripClusters                            = cms.InputTag("siStripClusters"),
    oldClusterRemovalInfo                    = cms.InputTag("mixedTripletStepClusters"),
    overrideTrkQuals                         = cms.InputTag('mixedTripletStep'),
    TrackQuality                             = cms.string('highPurity'),
    minNumberOfLayersWithMeasBeforeFiltering = cms.int32(0),
)

# SEEDING LAYERS
from RecoLocalTracker.SiStripClusterizer.SiStripClusterChargeCut_cfi import *
pixelLessStepSeedLayers = cms.EDProducer("SeedingLayersEDProducer",
    layerList = cms.vstring(
    #TIB
    'TIB1+TIB2+MTIB3',
    #TIB+TID
    'TIB1+TIB2+MTID1_pos','TIB1+TIB2+MTID1_neg',
    #TID
    'TID1_pos+TID2_pos+TID3_pos','TID1_neg+TID2_neg+TID3_neg',#ring 1-2 (matched)
    'TID1_pos+TID2_pos+MTID3_pos','TID1_neg+TID2_neg+MTID3_neg',#ring 3 (mono)
    'TID1_pos+TID2_pos+MTEC1_pos','TID1_neg+TID2_neg+MTEC1_neg',
    #TID+TEC RING 1-3
    'TID2_pos+TID3_pos+TEC1_pos','TID2_neg+TID3_neg+TEC1_neg',#ring 1-2 (matched)
    'TID2_pos+TID3_pos+MTEC1_pos','TID2_neg+TID3_neg+MTEC1_neg',#ring 3 (mono)
    #TEC RING 1-3
    'TEC1_pos+TEC2_pos+TEC3_pos', 'TEC1_neg+TEC2_neg+TEC3_neg',
    'TEC1_pos+TEC2_pos+MTEC3_pos','TEC1_neg+TEC2_neg+MTEC3_neg',
    'TEC1_pos+TEC2_pos+TEC4_pos', 'TEC1_neg+TEC2_neg+TEC4_neg',
    'TEC1_pos+TEC2_pos+MTEC4_pos','TEC1_neg+TEC2_neg+MTEC4_neg',
    'TEC2_pos+TEC3_pos+TEC4_pos', 'TEC2_neg+TEC3_neg+TEC4_neg',
    'TEC2_pos+TEC3_pos+MTEC4_pos','TEC2_neg+TEC3_neg+MTEC4_neg',
    'TEC2_pos+TEC3_pos+TEC5_pos', 'TEC2_neg+TEC3_neg+TEC5_neg',
    'TEC2_pos+TEC3_pos+TEC6_pos', 'TEC2_neg+TEC3_neg+TEC6_neg',
    'TEC3_pos+TEC4_pos+TEC5_pos', 'TEC3_neg+TEC4_neg+TEC5_neg',
    'TEC3_pos+TEC4_pos+MTEC5_pos','TEC3_neg+TEC4_neg+MTEC5_neg',
    'TEC3_pos+TEC5_pos+TEC6_pos', 'TEC3_neg+TEC5_neg+TEC6_neg',
    'TEC4_pos+TEC5_pos+TEC6_pos', 'TEC4_neg+TEC5_neg+TEC6_neg'    
    ),
    TIB = cms.PSet(
         TTRHBuilder    = cms.string('WithTrackAngle'), clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight')),
         matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
         skipClusters   = cms.InputTag('pixelLessStepClusters')
    ),
    MTIB = cms.PSet(
         TTRHBuilder    = cms.string('WithTrackAngle'), clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight')),
         skipClusters   = cms.InputTag('pixelLessStepClusters'),
         rphiRecHits    = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
    ),
    TID = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        skipClusters = cms.InputTag('pixelLessStepClusters'),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'), clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight')),
        minRing = cms.int32(1),
        maxRing = cms.int32(2)
    ),
    MTID = cms.PSet(
        rphiRecHits    = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
        skipClusters = cms.InputTag('pixelLessStepClusters'),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'), clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight')),
        minRing = cms.int32(3),
        maxRing = cms.int32(3)
    ),
    TEC = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        skipClusters = cms.InputTag('pixelLessStepClusters'),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'), clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight')),
        minRing = cms.int32(1),
        maxRing = cms.int32(2)
    ),
    MTEC = cms.PSet(
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
        skipClusters = cms.InputTag('pixelLessStepClusters'),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'), clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight')),
        minRing = cms.int32(3),
        maxRing = cms.int32(3)
    )
)

# SEEDS
import RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff
pixelLessStepSeeds = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone()
#OrderedHitsFactory
pixelLessStepSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'pixelLessStepSeedLayers'
pixelLessStepSeeds.OrderedHitsFactoryPSet.ComponentName = 'StandardMultiHitGenerator'
import RecoTracker.TkSeedGenerator.MultiHitGeneratorFromChi2_cfi
pixelLessStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet = RecoTracker.TkSeedGenerator.MultiHitGeneratorFromChi2_cfi.MultiHitGeneratorFromChi2.clone()
#SeedCreator
pixelLessStepSeeds.SeedCreatorPSet.ComponentName = 'SeedFromConsecutiveHitsTripletOnlyCreator'
#RegionFactory
pixelLessStepSeeds.RegionFactoryPSet.RegionPSet.ptMin = 0.4
pixelLessStepSeeds.RegionFactoryPSet.RegionPSet.originHalfLength = 12.0
pixelLessStepSeeds.RegionFactoryPSet.RegionPSet.originRadius = 1.0
#SeedComparitor
import RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi
pixelLessStepClusterShapeHitFilter  = RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi.ClusterShapeHitFilterESProducer.clone(
	ComponentName = cms.string('pixelLessStepClusterShapeHitFilter'),
        PixelShapeFile= cms.string('RecoPixelVertexing/PixelLowPtUtilities/data/pixelShape.par'),
        doStripShapeCut = cms.bool(False),
	clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight'))
	)
import RecoPixelVertexing.PixelLowPtUtilities.StripSubClusterShapeSeedFilter_cfi
pixelLessStepSeeds.SeedComparitorPSet = cms.PSet(
    ComponentName = cms.string('CombinedSeedComparitor'),
        mode = cms.string("and"),
        comparitors = cms.VPSet(
            cms.PSet(
                ComponentName = cms.string('PixelClusterShapeSeedComparitor'),
                FilterAtHelixStage = cms.bool(True),
                FilterPixelHits = cms.bool(False),
                FilterStripHits = cms.bool(True),
                ClusterShapeHitFilterName = cms.string('pixelLessStepClusterShapeHitFilter'),
                ClusterShapeCacheSrc = cms.InputTag("siPixelClusterShapeCache") # not really needed here since FilterPixelHits=False
            ), 
            RecoPixelVertexing.PixelLowPtUtilities.StripSubClusterShapeSeedFilter_cfi.StripSubClusterShapeSeedFilter.clone()
        )
    )

# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
pixelLessStepTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    maxLostHits = 0,
    minimumNumberOfHits = 4,
    minPt = 0.1
    )

import RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimatorESProducer_cfi
pixelLessStepChi2Est = RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimatorESProducer_cfi.Chi2ChargeMeasurementEstimator.clone(
    ComponentName = cms.string('pixelLessStepChi2Est'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(9.0),
    clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight'))
)

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
pixelLessStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    MeasurementTrackerName = '',
    trajectoryFilter = cms.PSet(refToPSet_ = cms.string('pixelLessStepTrajectoryFilter')),
    minNrOfHitsForRebuild = 4,
    maxCand = 2,
    alwaysUseInvalidHits = False,
    estimator = cms.string('pixelLessStepChi2Est'),
    maxDPhiForLooperReconstruction = cms.double(2.0),
    maxPtForLooperReconstruction = cms.double(0.7)
    )

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
pixelLessStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('pixelLessStepSeeds'),
    clustersToSkip = cms.InputTag('pixelLessStepClusters'),
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = cms.int32(50),
    #onlyPixelHitsForSeedCleaner = cms.bool(True),
    TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string('pixelLessStepTrajectoryBuilder'))
)

from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi import trajectoryCleanerBySharedHits
pixelLessStepTrajectoryCleanerBySharedHits = trajectoryCleanerBySharedHits.clone(
    ComponentName = cms.string('pixelLessStepTrajectoryCleanerBySharedHits'),
    fractionShared = cms.double(0.11),
    allowSharedFirstHit = cms.bool(True)
    )
pixelLessStepTrackCandidates.TrajectoryCleaner = 'pixelLessStepTrajectoryCleanerBySharedHits'


# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
pixelLessStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'pixelLessStepTrackCandidates',
    AlgorithmName = cms.string('pixelLessStep'),
    Fitter = cms.string('FlexibleKFFittingSmoother')
    )

import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
from RecoTracker.IterativeTracking.MixedTripletStep_cff import mixedTripletStepSelector
pixelLessStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src='pixelLessStepTracks',
    useAnyMVA = cms.bool(True),
    GBRForestLabel = cms.string('MVASelectorIter5_13TeV_v0'),
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'pixelLessStepLoose',
            chi2n_par = 9999,
            useMVA = cms.bool(True),
            minMVA = cms.double(0.4),
            #chi2n_par = 0.4,
            #res_par = ( 0.003, 0.001 ),
            #minNumberLayers = 4,
            #maxNumberLostLayers = 1,
            #minNumber3DLayers = 3,
            d0_par1 = ( 1.2, 4.0 ),
            dz_par1 = ( 1.2, 4.0 ),
            d0_par2 = ( 1.2, 4.0 ),
            dz_par2 = ( 1.2, 4.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'pixelLessStepTight',
            preFilterName = 'pixelLessStepLoose',
            chi2n_par = 0.3,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 4,
            maxNumberLostLayers = 0,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.9, 4.0 ),
            dz_par1 = ( 0.9, 4.0 ),
            d0_par2 = ( 0.9, 4.0 ),
            dz_par2 = ( 0.9, 4.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'pixelLessStep',
            preFilterName = 'pixelLessStepLoose',
            chi2n_par = cms.double(9999),
            useMVA = cms.bool(True),
            minMVA = cms.double(0.4),
            qualityBit = cms.string('highPurity'),
            keepAllTracks = cms.bool(True),
            #chi2n_par = 0.2,
            #res_par = ( 0.003, 0.001 ),
            #minNumberLayers = 4,
            #maxNumberLostLayers = 0,
            #minNumber3DLayers = 3,
            #max_minMissHitOutOrIn = 2,
            #max_lostHitFraction = 1.0,
            d0_par1 = ( 0.7, 4.0 ),
            dz_par1 = ( 0.7, 4.0 ),
            d0_par2 = ( 0.7, 4.0 ),
            dz_par2 = ( 0.7, 4.0 )
            ),
        mixedTripletStepSelector.trackSelectors[4].clone(
            name = 'pixelLessStepVtx',
            preFilterName=cms.string(''),
            keepAllTracks = cms.bool(False)
            ),
        mixedTripletStepSelector.trackSelectors[5].clone(
            name = 'pixelLessStepTrk',
            preFilterName=cms.string(''),
            keepAllTracks = cms.bool(False)
            )
        ) #end of vpset
    ) #end of clone

# need to merge the three sets
import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
pixelLessStep = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = cms.VInputTag(cms.InputTag("pixelLessStepTracks"),
                                   cms.InputTag("pixelLessStepTracks"),
                                   cms.InputTag("pixelLessStepTracks")),
    hasSelector=cms.vint32(1,1,1),
    shareFrac=cms.double(0.11),
    indivShareFrac=cms.vdouble(0.11,0.11,0.11),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("pixelLessStepSelector","pixelLessStep"),
                                       cms.InputTag("pixelLessStepSelector","pixelLessStepVtx"),
                                       cms.InputTag("pixelLessStepSelector","pixelLessStepTrk")),
    setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0,1,2), pQual=cms.bool(True) )),
    writeOnlyTrkQuals=cms.bool(True)
)                        

PixelLessStep = cms.Sequence(pixelLessStepClusters*
                             pixelLessStepSeedLayers*
                             pixelLessStepSeeds*
                             pixelLessStepTrackCandidates*
                             pixelLessStepTracks*
                             pixelLessStepSelector*
                             pixelLessStep)
