import FWCore.ParameterSet.Config as cms

# NEW CLUSTERS (remove previously used clusters)
lowPtTripletStepClusters = cms.EDProducer("TrackClusterRemover",
    clusterLessSolution= cms.bool(True),
    trajectories = cms.InputTag("lowPtQuadStepTracks"),
    overrideTrkQuals = cms.InputTag('lowPtQuadStepSelector','lowPtQuadStep'),
    TrackQuality = cms.string('highPurity'),
    minNumberOfLayersWithMeasBeforeFiltering = cms.int32(0),
    pixelClusters = cms.InputTag("siPixelClusters"),
    stripClusters = cms.InputTag("siStripClusters"),
    Common = cms.PSet(
        maxChi2 = cms.double(9.0)
    )
)

# SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi
lowPtTripletStepSeedLayers = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.PixelLayerTriplets.clone(
    layerList = cms.vstring('BPix1+BPix2+BPix3', 'BPix2+BPix3+BPix4',
                            'BPix1+BPix3+BPix4', 'BPix1+BPix2+BPix4',
                            'BPix2+BPix3+FPix1_pos', 'BPix2+BPix3+FPix1_neg',
                            'BPix1+BPix2+FPix1_pos', 'BPix1+BPix2+FPix1_neg',
                            'BPix1+BPix3+FPix1_pos', 'BPix1+BPix3+FPix1_neg',
                            'BPix2+FPix1_pos+FPix2_pos', 'BPix2+FPix1_neg+FPix2_neg',
                            'BPix1+FPix1_pos+FPix2_pos', 'BPix1+FPix1_neg+FPix2_neg',
                            'BPix1+BPix2+FPix2_pos', 'BPix1+BPix2+FPix2_neg',
                            'FPix1_pos+FPix2_pos+FPix3_pos', 'FPix1_neg+FPix2_neg+FPix3_neg',
                            'BPix1+FPix2_pos+FPix3_pos', 'BPix1+FPix2_neg+FPix3_neg',
                            'BPix1+FPix1_pos+FPix3_pos', 'BPix1+FPix1_neg+FPix3_neg')
    )
lowPtTripletStepSeedLayers.BPix.skipClusters = cms.InputTag('lowPtTripletStepClusters')
lowPtTripletStepSeedLayers.FPix.skipClusters = cms.InputTag('lowPtTripletStepClusters')

# SEEDS
import RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff
from RecoTracker.TkTrackingRegions.GlobalTrackingRegionFromBeamSpot_cfi import RegionPsetFomBeamSpotBlock
lowPtTripletStepSeeds = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone(
    RegionFactoryPSet = RegionPsetFomBeamSpotBlock.clone(
    ComponentName = cms.string('GlobalRegionProducerFromBeamSpot'),
    RegionPSet = RegionPsetFomBeamSpotBlock.RegionPSet.clone(
    ptMin = 0.35,
    originRadius = 0.015,
    nSigmaZ = 4.0
    )
    )
    )
lowPtTripletStepSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'lowPtTripletStepSeedLayers'

from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
lowPtTripletStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.SeedComparitorPSet.ComponentName = 'LowPtClusterShapeSeedComparitor'
lowPtTripletStepSeeds.ClusterCheckPSet.doClusterCheck = cms.bool(False)
lowPtTripletStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = cms.uint32(0)

# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi
lowPtTripletStepStandardTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone(
    ComponentName = 'lowPtTripletStepStandardTrajectoryFilter',
    filterPset = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.filterPset.clone(
    minimumNumberOfHits = 3,
    minPt = 0.075
    )
    )

from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeTrajectoryFilterESProducer_cfi import *
# Composite filter
import TrackingTools.TrajectoryFiltering.CompositeTrajectoryFilterESProducer_cfi
lowPtTripletStepTrajectoryFilter = TrackingTools.TrajectoryFiltering.CompositeTrajectoryFilterESProducer_cfi.compositeTrajectoryFilterESProducer.clone(
    ComponentName = cms.string('lowPtTripletStepTrajectoryFilter'),
    filterNames   = cms.vstring('lowPtTripletStepStandardTrajectoryFilter',
                                'clusterShapeTrajectoryFilter')
    )

import TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi
lowPtTripletStepChi2Est = TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi.Chi2MeasurementEstimator.clone(
    ComponentName = cms.string('lowPtTripletStepChi2Est'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(9.0)
)

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
lowPtTripletStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone(
    ComponentName = 'lowPtTripletStepTrajectoryBuilder',
    MeasurementTrackerName = '',
    trajectoryFilterName = 'lowPtTripletStepTrajectoryFilter',
    clustersToSkip = cms.InputTag('lowPtTripletStepClusters'),
    maxCand = 4,
    estimator = cms.string('lowPtTripletStepChi2Est'),
    maxDPhiForLooperReconstruction = cms.double(2.0),
    # 0.63 GeV is the maximum pT for a charged particle to loop within the 1.1m radius
    # of the outermost Tracker barrel layer (with B=3.8T)
    maxPtForLooperReconstruction = cms.double(0.7) 
    )

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
lowPtTripletStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('lowPtTripletStepSeeds'),
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = cms.int32(50),
    onlyPixelHitsForSeedCleaner = cms.bool(True),
    TrajectoryBuilder = 'lowPtTripletStepTrajectoryBuilder',
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True
)

# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
lowPtTripletStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'lowPtTripletStepTrackCandidates',
    AlgorithmName = cms.string('iter3'),
    Fitter = cms.string('FlexibleKFFittingSmoother'),
    TTRHBuilder=cms.string('WithTrackAngle')
    )

from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi import trajectoryCleanerBySharedHits
lowPtTripletStepTrajectoryCleanerBySharedHits = trajectoryCleanerBySharedHits.clone(
        ComponentName = cms.string('lowPtTripletStepTrajectoryCleanerBySharedHits'),
            fractionShared = cms.double(0.09),
            allowSharedFirstHit = cms.bool(True)
            )
lowPtTripletStepTrackCandidates.TrajectoryCleaner = 'lowPtTripletStepTrajectoryCleanerBySharedHits'

# Final selection
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
lowPtTripletStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src='lowPtTripletStepTracks',
    trackSelectors= cms.VPSet(
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
        ) #end of vpset
    ) #end of clone

# Final sequence
LowPtTripletStep = cms.Sequence(lowPtTripletStepClusters*
                                lowPtTripletStepSeedLayers*
                                lowPtTripletStepSeeds*
                                lowPtTripletStepTrackCandidates*
                                lowPtTripletStepTracks*
                                lowPtTripletStepSelector)
