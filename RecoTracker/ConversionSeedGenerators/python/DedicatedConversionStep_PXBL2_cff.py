import FWCore.ParameterSet.Config as cms

###########################################################################
# dedicated tracking step to improve conversion reconstruction efficiency #
###########################################################################

#PXL CONVERSIONS: SIXTH STEP

# REMOVE HITS ASSIGNED TO GOOD TRACKS FROM PREVIOUS ITERATIONS
pxbL2_Filter = cms.EDProducer("QualityFilter",
                              TrackQuality = cms.string('highPurity'),
                              recTracks = cms.InputTag("tobtecStep")
                              )

pxbL2_Clusters = cms.EDProducer("TrackClusterRemover",
                                oldClusterRemovalInfo = cms.InputTag("fifthClusters"),
                                trajectories = cms.InputTag("pxbL2_Filter"),
                                pixelClusters = cms.InputTag("previousStepPixelClusters"),
                                stripClusters = cms.InputTag("previousStepStripClusters"),
                                Common = cms.PSet(
                                    maxChi2 = cms.double(30.0)
                                    )
                                )

# TRACKER HITS
import RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi
pxbL2_PixelRecHits = RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi.siPixelRecHits.clone(
    src = 'pxbL2_Clusters'
    )
import RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi
pxbL2_StripRecHits = RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi.siStripMatchedRecHits.clone(
    ClusterProducer = 'pxbL2_Clusters'
    )

## Propagator taking into account momentum uncertainty in multiple scattering calculation.
#import TrackingTools.MaterialEffects.MaterialPropagator_cfi
#MaterialPropagatorPtMin01 = TrackingTools.MaterialEffects.MaterialPropagator_cfi.MaterialPropagator.clone(
#    ComponentName = 'PropagatorWithMaterialPtMin01',
#    ptMin = 0.1
#    )
#
#import TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi
#OppositeMaterialPropagatorPtMin01 = TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi.OppositeMaterialPropagator.clone(
#    ComponentName = 'PropagatorWithMaterialOppositePtMin01',
#    ptMin = 0.1
#    )
# SEEDING LAYERS
pxbL2_layertriplets = cms.ESProducer("SeedingLayersESProducer",
    ComponentName = cms.string('PxbL2_LayerTriplets'),
    layerList = cms.vstring('BPix2+BPix3+TIB1', 
        'BPix2+BPix3+TIB2','BPix3+TIB1+TIB2'),
    BPix = cms.PSet(
        useErrorsFromParam = cms.bool(True),
        hitErrorRPhi = cms.double(0.0027),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedTriplets'),
        HitProducer = cms.string('pxbL2_PixelRecHits'),
        hitErrorRZ = cms.double(0.006)
    ),
    TIB = cms.PSet(
        matchedRecHits = cms.InputTag("pxbL2_StripRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    )
)
# SEEDS
from RecoPixelVertexing.PixelTriplets.PixelTripletLargeTipGenerator_cfi import *
PixelTripletLargeTipGenerator.extraHitRZtolerance = 0.0
PixelTripletLargeTipGenerator.extraHitRPhitolerance = 0.0
import RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff
pxbL2_Triplets = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone()
pxbL2_Triplets.OrderedHitsFactoryPSet.SeedingLayers = 'PxbL2_LayerTriplets'
pxbL2_Triplets.OrderedHitsFactoryPSet.GeneratorPSet = cms.PSet(PixelTripletLargeTipGenerator)
pxbL2_Triplets.SeedCreatorPSet.ComponentName = 'SeedFromConsecutiveHitsTripletOnlyCreator'
pxbL2_Triplets.RegionFactoryPSet.RegionPSet.ptMin = 0.1
pxbL2_Triplets.RegionFactoryPSet.RegionPSet.originHalfLength = 2.0
pxbL2_Triplets.RegionFactoryPSet.RegionPSet.originRadius = 25.0
pxbL2_SeedsPositive = cms.EDFilter("SeedChargeSelector",src=cms.InputTag("pxbL2_Triplets"),charge = cms.int32(1))
pxbL2_SeedsNegative = cms.EDFilter("SeedChargeSelector",src=cms.InputTag("pxbL2_Triplets"),charge = cms.int32(-1))
pxbL2_Seeds =  cms.EDProducer("ConversionSeedFilterCharge",
                             seedCollectionPos = cms.InputTag("pxbL2_SeedsPositive"),
                             seedCollectionNeg = cms.InputTag("pxbL2_SeedsNegative"),
                             deltaPhiCut = cms.double(1.5),
                             deltaCotThetaCut = cms.double(0.25),
                             deltaRCut = cms.double(5.),
                             deltaZCut = cms.double(5.),
                             maxInputSeeds = cms.uint32(200)
                             )
#pxbL2_Seeds =  cms.EDProducer("ConversionSeedFilterFwk",
#                             src = cms.InputTag("pxbL2_Triplets"),
#pxbL2_Seeds =  cms.EDProducer("ConversionSeedFilter",
#                             seedCollection = cms.InputTag("pxbL2_Triplets"),
#                             chargeCut = cms.bool(True),
#                             deltaPhiCut = cms.double(1.5),
#                             deltaCotThetaCut = cms.double(0.25),
#                             deltaRCut = cms.double(5.),
#                             deltaZCut = cms.double(5.)
#                             )
# TRACKER DATA CONTROL
import RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi
pxbL2_MeasurementTracker = RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi.MeasurementTracker.clone(
    ComponentName = 'pxbL2_MeasurementTracker',
    pixelClusterProducer = 'pxbL2_Clusters',
    stripClusterProducer = 'pxbL2_Clusters'
    )
# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi
pxbL2_CkfTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone(
    ComponentName = 'pxbL2_CkfTrajectoryFilter',
    filterPset = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.filterPset.clone(
    maxLostHits = 1,
    minimumNumberOfHits = 3,
    minPt = 0.09
    )
    )
# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
pxbL2_CkfTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone(
    ComponentName = 'pxbL2_CkfTrajectoryBuilder',
    MeasurementTrackerName = 'pxbL2_MeasurementTracker',
    trajectoryFilterName = 'pxbL2_CkfTrajectoryFilter',
    propagatorAlong = cms.string('PropagatorWithMaterialPtMin01'),
    propagatorOpposite = cms.string('PropagatorWithMaterialOppositePtMin01')
    )
# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
pxbL2_TrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('pxbL2_Seeds'),
    TrajectoryBuilder = 'pxbL2_CkfTrajectoryBuilder',
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True
)
# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
pxbL2_WithMaterialTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    AlgorithmName = cms.string('iter6'),
    src = 'pxbL2_TrackCandidates',
    clusterRemovalInfo = 'pxbL2_Clusters',
)
# TRACK SELECTION AND QUALITY FLAG SETTING.
import RecoTracker.FinalTrackSelectors.selectLoose_cfi
import RecoTracker.FinalTrackSelectors.selectTight_cfi
import RecoTracker.FinalTrackSelectors.selectHighPurity_cfi
import RecoTracker.FinalTrackSelectors.simpleTrackListMerger_cfi
pxbL2_StepLoose = RecoTracker.FinalTrackSelectors.selectLoose_cfi.selectLoose.clone(
    src = 'pxbL2_WithMaterialTracks',
    keepAllTracks = True,
    copyExtras = False,
    copyTrajectories = True,
    chi2n_par = 2.,
    res_par = ( 0.003, 0.001 ),
    minNumberLayers = 3,
    maxNumberLostLayers = 1,
    minNumber3DLayers = 1,
    d0_par1 = ( 5., 8.0 ),
    dz_par1 = ( 5., 8.0 ),
    d0_par2 = ( 5., 8.0 ),
    dz_par2 = ( 5., 8.0 )
    )
pxbL2_StepTight = RecoTracker.FinalTrackSelectors.selectTight_cfi.selectTight.clone(
    src = 'pxbL2_StepLoose',
    keepAllTracks = True,
    copyExtras = False,
    copyTrajectories = True,
    chi2n_par = 2.,
    res_par = ( 0.003, 0.001 ),
    minNumberLayers = 3,
    maxNumberLostLayers = 1,
    minNumber3DLayers = 1,
    d0_par1 = ( 5., 8.0 ),
    dz_par1 = ( 5., 8.0 ),
    d0_par2 = ( 5., 8.0 ),
    dz_par2 = ( 5., 8.0 )
    )
pxbL2_Step = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone(
    src = 'pxbL2_StepTight',
    keepAllTracks = True,
    copyExtras = False,
    copyTrajectories = True,
    chi2n_par = 2.,
    res_par = ( 0.003, 0.001 ),
    minNumberLayers = 3,
    maxNumberLostLayers = 1,
    minNumber3DLayers = 1,
    d0_par1 = ( 5., 8.0 ),
    dz_par1 = ( 5., 8.0 ),
    d0_par2 = ( 5., 8.0 ),
    dz_par2 = ( 5., 8.0 )
    )

