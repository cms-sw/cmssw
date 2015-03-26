import FWCore.ParameterSet.Config as cms


# NEW CLUSTERS (remove previously used clusters)
from RecoLocalTracker.SubCollectionProducers.trackClusterRemover_cfi import *
pixelPairStepClusters = trackClusterRemover.clone(
    maxChi2                                  = cms.double(9.0),
    trajectories                             = cms.InputTag("lowPtTripletStepTracks"),
    pixelClusters                            = cms.InputTag("siPixelClusters"),
    stripClusters                            = cms.InputTag("siStripClusters"),
    oldClusterRemovalInfo                    = cms.InputTag("lowPtTripletStepClusters"),
    overrideTrkQuals                         = cms.InputTag('lowPtTripletStepSelector','lowPtTripletStep'),
    TrackQuality                             = cms.string('highPurity'),
    minNumberOfLayersWithMeasBeforeFiltering = cms.int32(0),
)

# SEEDING LAYERS
pixelPairStepSeedLayers = cms.EDProducer("SeedingLayersEDProducer",
    layerList = cms.vstring('BPix1+BPix2', 'BPix1+BPix3', 'BPix2+BPix3', 
        'BPix1+FPix1_pos', 'BPix1+FPix1_neg', 
        'BPix2+FPix1_pos', 'BPix2+FPix1_neg', 
        'BPix1+FPix2_pos', 'BPix1+FPix2_neg', 
        'FPix1_pos+FPix2_pos', 'FPix1_neg+FPix2_neg'),
    BPix = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        HitProducer = cms.string('siPixelRecHits'),
        skipClusters = cms.InputTag('pixelPairStepClusters')
    ),
    FPix = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        HitProducer = cms.string('siPixelRecHits'),
        skipClusters = cms.InputTag('pixelPairStepClusters')
    )
)

# SEEDS
import RecoTracker.TkSeedGenerator.GlobalSeedsFromPairsWithVertices_cff
pixelPairStepSeeds = RecoTracker.TkSeedGenerator.GlobalSeedsFromPairsWithVertices_cff.globalSeedsFromPairsWithVertices.clone()
pixelPairStepSeeds.RegionFactoryPSet.RegionPSet.ptMin = 0.3
pixelPairStepSeeds.RegionFactoryPSet.RegionPSet.originRadius = 0.015
pixelPairStepSeeds.RegionFactoryPSet.RegionPSet.fixedError = 0.03
pixelPairStepSeeds.OrderedHitsFactoryPSet.SeedingLayers = cms.InputTag('pixelPairStepSeedLayers')

# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
pixelPairStepTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    minimumNumberOfHits = 3,
    minPt = 0.05
    )

import TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi
pixelPairStepChi2Est = TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi.Chi2MeasurementEstimator.clone(
    ComponentName = cms.string('pixelPairStepChi2Est'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(25.0)
)

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
pixelPairStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    MeasurementTrackerName = '',
    trajectoryFilter = cms.PSet(refToPSet_ = cms.string('pixelPairStepTrajectoryFilter')),
    clustersToSkip = cms.InputTag('pixelPairStepClusters'),
    maxCand = 6,
    estimator = cms.string('pixelPairStepChi2Est'),
    maxDPhiForLooperReconstruction = cms.double(2.0),
    maxPtForLooperReconstruction = cms.double(0.7) 
    )

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
pixelPairStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('pixelPairStepSeeds'),
    TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string('pixelPairStepTrajectoryBuilder')),
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = cms.int32(50),
    onlyPixelHitsForSeedCleaner = cms.bool(True),

)

from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi import trajectoryCleanerBySharedHits
pixelPairStepTrajectoryCleanerBySharedHits = trajectoryCleanerBySharedHits.clone(
    ComponentName = cms.string('pixelPairStepTrajectoryCleanerBySharedHits'),
    fractionShared = cms.double(0.19),
    allowSharedFirstHit = cms.bool(True)
    )
pixelPairStepTrackCandidates.TrajectoryCleaner = 'pixelPairStepTrajectoryCleanerBySharedHits'


# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
pixelPairStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    AlgorithmName = cms.string('pixelPairStep'),
    src = 'pixelPairStepTrackCandidates',
    Fitter = cms.string('FlexibleKFFittingSmoother')
    )

# Final selection
import RecoTracker.IterativeTracking.LowPtTripletStep_cff
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
pixelPairStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src='pixelPairStepTracks',
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'pixelPairStepLoose',
            chi2n_par = 1.0,
            res_par = ( 0.003, 0.002 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 3,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.55, 4.0 ),
            dz_par1 = ( 0.55, 4.0 ),
            d0_par2 = ( 0.5, 4.0 ),
            dz_par2 = ( 0.5, 4.0 )
            ), #end of pset
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'pixelPairStepTight',
            preFilterName = 'pixelPairStepLoose',
            chi2n_par = 0.8,
            res_par = ( 0.003, 0.002 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 2,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.5, 4.0 ),
            dz_par1 = ( 0.5, 4.0 ),
            d0_par2 = ( 0.4, 4.0 ),
            dz_par2 = ( 0.4, 4.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'pixelPairStep',
            preFilterName = 'pixelPairStepTight',
            chi2n_par = 0.6,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 2,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.45, 4.0 ),
            dz_par1 = ( 0.45, 4.0 ),
            d0_par2 = ( 0.3, 4.0 ),
            dz_par2 = ( 0.35, 4.0 )
            ),
        ) #end of vpset
    ) #end of clone

# Final sequence
PixelPairStep = cms.Sequence(pixelPairStepClusters*
                         pixelPairStepSeedLayers*
                         pixelPairStepSeeds*
                         pixelPairStepTrackCandidates*
                         pixelPairStepTracks*
                         pixelPairStepSelector)
