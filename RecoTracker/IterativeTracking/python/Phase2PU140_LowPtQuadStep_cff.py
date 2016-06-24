import FWCore.ParameterSet.Config as cms

# NEW CLUSTERS (remove previously used clusters)
from RecoLocalTracker.SubCollectionProducers.phase2trackClusterRemover_cfi import *
lowPtQuadStepClusters = phase2trackClusterRemover.clone(
    maxChi2                                  = cms.double(9.0),
    trajectories                             = cms.InputTag("highPtTripletStepTracks"),
    phase2pixelClusters                      = cms.InputTag("siPixelClusters"),
    phase2OTClusters                         = cms.InputTag("siPhase2Clusters"),
    oldClusterRemovalInfo                    = cms.InputTag("highPtTripletStepClusters"),
    overrideTrkQuals                         = cms.InputTag('highPtTripletStepSelector','highPtTripletStep'),
    TrackQuality                             = cms.string('highPurity'),
    minNumberOfLayersWithMeasBeforeFiltering = cms.int32(0),
)

# SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi
lowPtQuadStepSeedLayers = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.PixelLayerTriplets.clone(
    layerList = cms.vstring('BPix1+BPix2+BPix3', 'BPix2+BPix3+BPix4',
                            'BPix2+BPix3+FPix1_pos', 'BPix2+BPix3+FPix1_neg',
                            'BPix1+BPix2+FPix1_pos', 'BPix1+BPix2+FPix1_neg',
                            'BPix2+FPix1_pos+FPix2_pos', 'BPix2+FPix1_neg+FPix2_neg',
                            'BPix1+FPix1_pos+FPix2_pos', 'BPix1+FPix1_neg+FPix2_neg',
                            'BPix1+FPix2_pos+FPix3_pos', 'BPix1+FPix2_neg+FPix3_neg',
                            'FPix1_pos+FPix2_pos+FPix3_pos', 'FPix1_neg+FPix2_neg+FPix3_neg',
                            'FPix2_pos+FPix3_pos+FPix4_pos', 'FPix2_neg+FPix3_neg+FPix4_neg',
                            'FPix3_pos+FPix4_pos+FPix5_pos', 'FPix3_neg+FPix4_neg+FPix5_neg',
                            'FPix4_pos+FPix5_pos+FPix6_pos', 'FPix4_neg+FPix5_neg+FPix6_neg',
                            'FPix5_pos+FPix6_pos+FPix7_pos', 'FPix5_neg+FPix6_neg+FPix7_neg',
                            'FPix6_pos+FPix7_pos+FPix8_pos', 'FPix6_neg+FPix7_neg+FPix8_neg',
                            'FPix6_pos+FPix7_pos+FPix9_pos', 'FPix6_neg+FPix7_neg+FPix9_neg')
    )
lowPtQuadStepSeedLayers.BPix.skipClusters = cms.InputTag('lowPtQuadStepClusters')
lowPtQuadStepSeedLayers.FPix.skipClusters = cms.InputTag('lowPtQuadStepClusters')

# SEEDS
import RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff
from RecoTracker.TkTrackingRegions.GlobalTrackingRegionFromBeamSpot_cfi import RegionPsetFomBeamSpotBlock
from RecoPixelVertexing.PixelTriplets.quadrupletseedmerging_cff import *
lowPtQuadStepSeeds = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone(
    RegionFactoryPSet = RegionPsetFomBeamSpotBlock.clone(
        ComponentName = cms.string('GlobalRegionProducerFromBeamSpot'),
        RegionPSet = RegionPsetFomBeamSpotBlock.RegionPSet.clone(
            ptMin = 0.35,
            originRadius = 0.02,
            nSigmaZ = 4.0
            )
    ),
    SeedMergerPSet = cms.PSet(
        layerList = cms.PSet(refToPSet_ = cms.string("PixelSeedMergerQuadruplets")),
	addRemainingTriplets = cms.bool(False),
	mergeTriplets = cms.bool(True),
	ttrhBuilderLabel = cms.string('PixelTTRHBuilderWithoutAngle')
    )
)
lowPtQuadStepSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'lowPtQuadStepSeedLayers'
lowPtQuadStepSeeds.SeedCreatorPSet.magneticField = ''
lowPtQuadStepSeeds.SeedCreatorPSet.propagator = 'PropagatorWithMaterial'

from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
import RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi
lowPtQuadStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.SeedComparitorPSet = RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi.LowPtClusterShapeSeedComparitor
lowPtQuadStepSeeds.ClusterCheckPSet.doClusterCheck = cms.bool(False)
lowPtQuadStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = cms.uint32(0)

# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
lowPtQuadStepStandardTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    minimumNumberOfHits = 3,
    minPt = 0.075
    )

from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeTrajectoryFilter_cfi import *
# Composite filter
lowPtQuadStepTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CompositeTrajectoryFilter_block.clone(
    filters   = [cms.PSet(refToPSet_ = cms.string('lowPtQuadStepStandardTrajectoryFilter')),
                 cms.PSet(refToPSet_ = cms.string('ClusterShapeTrajectoryFilter'))]
    )

import TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi
lowPtQuadStepChi2Est = TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi.Chi2MeasurementEstimator.clone(
    ComponentName = cms.string('lowPtQuadStepChi2Est'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(25.0)
)

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
lowPtQuadStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    MeasurementTrackerName = '',
    trajectoryFilter = cms.PSet(refToPSet_ = cms.string('lowPtQuadStepTrajectoryFilter')),
    #minNrOfHitsForRebuild = 1,
    maxCand = 5,
    estimator = cms.string('lowPtQuadStepChi2Est'),
    maxDPhiForLooperReconstruction = cms.double(2.0),
    # 0.63 GeV is the maximum pT for a charged particle to loop within the 1.1m radius
    # of the outermost Tracker barrel layer (with B=3.8T)
    maxPtForLooperReconstruction = cms.double(0.7) 
    )

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
lowPtQuadStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('lowPtQuadStepSeeds'),
    phase2clustersToSkip = cms.InputTag('lowPtQuadStepClusters'),
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = cms.int32(50),
    onlyPixelHitsForSeedCleaner = cms.bool(True),
    TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string('lowPtQuadStepTrajectoryBuilder')),
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True
)

# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
lowPtQuadStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'lowPtQuadStepTrackCandidates',
    AlgorithmName = cms.string('lowPtQuadStep'),
    Fitter = cms.string('FlexibleKFFittingSmoother'),
    TTRHBuilder=cms.string('WithTrackAngle')
    )

from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi import trajectoryCleanerBySharedHits
lowPtQuadStepTrajectoryCleanerBySharedHits = trajectoryCleanerBySharedHits.clone(
        ComponentName = cms.string('lowPtQuadStepTrajectoryCleanerBySharedHits'),
            fractionShared = cms.double(0.09),
            allowSharedFirstHit = cms.bool(True)
            )
lowPtQuadStepTrackCandidates.TrajectoryCleaner = 'lowPtQuadStepTrajectoryCleanerBySharedHits'

# Final selection
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
lowPtQuadStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src='lowPtQuadStepTracks',
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
        ) #end of vpset
    ) #end of clone

# Final sequence
LowPtQuadStep = cms.Sequence(lowPtQuadStepClusters*
                                lowPtQuadStepSeedLayers*
                                lowPtQuadStepSeeds*
                                lowPtQuadStepTrackCandidates*
                                lowPtQuadStepTracks*
                                lowPtQuadStepSelector)
