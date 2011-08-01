import FWCore.ParameterSet.Config as cms

###########################################################################
# dedicated tracking step to improve conversion reconstruction efficiency #
###########################################################################

#PXL CONVERSIONS: SIXTH STEP

# REMOVE HITS ASSIGNED TO GOOD TRACKS FROM PREVIOUS ITERATIONS
fifthFilter = cms.EDProducer("QualityFilter",
    TrackQuality = cms.string('highPurity'),
    recTracks = cms.InputTag("tobtecStep")
)
sixthClusters = cms.EDProducer("TrackClusterRemover",
    oldClusterRemovalInfo = cms.InputTag("fifthClusters"),
    trajectories = cms.InputTag("fifthFilter"),
    pixelClusters = cms.InputTag("fifthClusters"),
    stripClusters = cms.InputTag("fifthClusters"),
    Common = cms.PSet(
        maxChi2 = cms.double(30.0)
    )
)
# TRACKER HITS
import RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi
sixthPixelRecHits = RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi.siPixelRecHits.clone(
    src = 'sixthClusters'
    )
import RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi
sixthStripRecHits = RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi.siStripMatchedRecHits.clone(
    ClusterProducer = 'sixthClusters'
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
sixthlayertriplets = cms.ESProducer("SeedingLayersESProducer",
    ComponentName = cms.string('SixthLayerTriplets'),
    layerList = cms.vstring('BPix2+BPix3+TIB1', 
        'BPix2+BPix3+TIB2','BPix3+TIB1+TIB2'),
    BPix = cms.PSet(
        useErrorsFromParam = cms.bool(True),
        hitErrorRPhi = cms.double(0.0027),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedTriplets'),
        HitProducer = cms.string('sixthPixelRecHits'),
        hitErrorRZ = cms.double(0.006)
    ),
    TIB = cms.PSet(
        matchedRecHits = cms.InputTag("sixthStripRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    )
)
# SEEDS
from RecoPixelVertexing.PixelTriplets.PixelTripletLargeTipGenerator_cfi import *
PixelTripletLargeTipGenerator.extraHitRZtolerance = 0.0
PixelTripletLargeTipGenerator.extraHitRPhitolerance = 0.0
import RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff
sixthTriplets = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone()
sixthTriplets.OrderedHitsFactoryPSet.SeedingLayers = 'SixthLayerTriplets'
sixthTriplets.OrderedHitsFactoryPSet.GeneratorPSet = cms.PSet(PixelTripletLargeTipGenerator)
sixthTriplets.SeedCreatorPSet.ComponentName = 'SeedFromConsecutiveHitsTripletOnlyCreator'
sixthTriplets.RegionFactoryPSet.RegionPSet.ptMin = 0.1
sixthTriplets.RegionFactoryPSet.RegionPSet.originHalfLength = 2.0
sixthTriplets.RegionFactoryPSet.RegionPSet.originRadius = 25.0
sixthSeedsPositive = cms.EDFilter("SeedChargeSelector",src=cms.InputTag("sixthTriplets"),charge = cms.int32(1))
sixthSeedsNegative = cms.EDFilter("SeedChargeSelector",src=cms.InputTag("sixthTriplets"),charge = cms.int32(-1))
sixthSeeds =  cms.EDProducer("ConversionSeedFilterCharge",
                             seedCollectionPos = cms.InputTag("sixthSeedsPositive"),
                             seedCollectionNeg = cms.InputTag("sixthSeedsNegative"),
                             deltaPhiCut = cms.double(1.5),
                             deltaCotThetaCut = cms.double(0.25),
                             deltaRCut = cms.double(5.),
                             deltaZCut = cms.double(5.),
                             maxInputSeeds = cms.uint32(200)
                             )
#sixthSeeds =  cms.EDProducer("ConversionSeedFilterFwk",
#                             src = cms.InputTag("sixthTriplets"),
#sixthSeeds =  cms.EDProducer("ConversionSeedFilter",
#                             seedCollection = cms.InputTag("sixthTriplets"),
#                             chargeCut = cms.bool(True),
#                             deltaPhiCut = cms.double(1.5),
#                             deltaCotThetaCut = cms.double(0.25),
#                             deltaRCut = cms.double(5.),
#                             deltaZCut = cms.double(5.)
#                             )
# TRACKER DATA CONTROL
import RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi
sixthMeasurementTracker = RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi.MeasurementTracker.clone(
    ComponentName = 'sixthMeasurementTracker',
    pixelClusterProducer = 'sixthClusters',
    stripClusterProducer = 'sixthClusters'
    )
# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi
sixthCkfTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone(
    ComponentName = 'sixthCkfTrajectoryFilter',
    filterPset = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.filterPset.clone(
    maxLostHits = 1,
    minimumNumberOfHits = 3,
    minPt = 0.09
    )
    )
# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
sixthCkfTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone(
    ComponentName = 'sixthCkfTrajectoryBuilder',
    MeasurementTrackerName = 'sixthMeasurementTracker',
    trajectoryFilterName = 'sixthCkfTrajectoryFilter',
    propagatorAlong = cms.string('PropagatorWithMaterialPtMin01'),
    propagatorOpposite = cms.string('PropagatorWithMaterialOppositePtMin01')
    )
# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
sixthTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('sixthSeeds'),
    TrajectoryBuilder = 'sixthCkfTrajectoryBuilder',
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True
)
# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
sixthWithMaterialTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    AlgorithmName = cms.string('iter6'),
    src = 'sixthTrackCandidates',
    clusterRemovalInfo = 'sixthClusters',
)
# TRACK SELECTION AND QUALITY FLAG SETTING.
import RecoTracker.FinalTrackSelectors.selectLoose_cfi
import RecoTracker.FinalTrackSelectors.selectTight_cfi
import RecoTracker.FinalTrackSelectors.selectHighPurity_cfi
import RecoTracker.FinalTrackSelectors.simpleTrackListMerger_cfi
sixthStepLoose = RecoTracker.FinalTrackSelectors.selectLoose_cfi.selectLoose.clone(
    src = 'sixthWithMaterialTracks',
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
sixthStepTight = RecoTracker.FinalTrackSelectors.selectTight_cfi.selectTight.clone(
    src = 'sixthStepLoose',
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
sixthStep = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone(
    src = 'sixthStepTight',
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

#PL CONVERSIONS: SEVENTH STEP
sixthFilter = cms.EDProducer("QualityFilter",
    TrackQuality = cms.string('highPurity'),
    recTracks = cms.InputTag("sixthStep")
)
seventhClusters = cms.EDProducer("TrackClusterRemover",
    oldClusterRemovalInfo = cms.InputTag("sixthClusters"),
    trajectories = cms.InputTag("sixthFilter"),
    pixelClusters = cms.InputTag("sixthClusters"),
    stripClusters = cms.InputTag("sixthClusters"),
    Common = cms.PSet(
        maxChi2 = cms.double(30.0)
    )
)
# TRACKER HITS
import RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi
import RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi
seventhPixelRecHits = RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi.siPixelRecHits.clone(
    src = 'seventhClusters'
    )
seventhStripRecHits = RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi.siStripMatchedRecHits.clone(
    ClusterProducer = 'seventhClusters'
    )
# SEEDING LAYERS
seventhlayerpairs = cms.ESProducer("SeedingLayersESProducer",
    ComponentName = cms.string('SeventhLayerPairs'),
    layerList = cms.vstring('TIB1+TIB2','TOB1+TOB2','TIB2+TOB1',
                            'TIB1+TID1_pos','TIB1+TID1_neg',
                            'TID1_pos+TID2_pos','TID2_pos+TID3_pos',
                            'TID1_neg+TID2_neg','TID2_neg+TID3_neg'),
    TOB = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        matchedRecHits = cms.InputTag("seventhStripRecHits","matchedRecHit")
    ),
    TIB = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        matchedRecHits = cms.InputTag("seventhStripRecHits","matchedRecHit")
    ),
    TID = cms.PSet(
        matchedRecHits = cms.InputTag("seventhStripRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        minRing = cms.int32(1),
        maxRing = cms.int32(2)
    ),
    TEC = cms.PSet(
        matchedRecHits = cms.InputTag("seventhStripRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        minRing = cms.int32(1),
        maxRing = cms.int32(2)
    )
)
# SEEDS
import RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cff
seventhPLSeeds = RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cff.globalMixedSeeds.clone()
import RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi
seventhPLSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'SeventhLayerPairs'
seventhPLSeeds.RegionFactoryPSet.RegionPSet.ptMin = 0.2
seventhPLSeeds.RegionFactoryPSet.RegionPSet.originHalfLength = 2.0
seventhPLSeeds.RegionFactoryPSet.RegionPSet.originRadius = 25.0
seventhSeedsPositive = cms.EDFilter("SeedChargeSelector",src=cms.InputTag("seventhPLSeeds"),charge = cms.int32(1))
seventhSeedsNegative = cms.EDFilter("SeedChargeSelector",src=cms.InputTag("seventhPLSeeds"),charge = cms.int32(-1))
seventhSeedsWithSeeds =  cms.EDProducer("ConversionSeedFilterCharge",
                                   seedCollectionPos = cms.InputTag("seventhSeedsPositive"),
                                   seedCollectionNeg = cms.InputTag("seventhSeedsNegative"),
                                   deltaPhiCut = cms.double(1.5),
                                   deltaCotThetaCut = cms.double(0.25),
                                   deltaRCut = cms.double(5.),
                                   deltaZCut = cms.double(5.),
                                   maxInputSeeds = cms.uint32(200)
                                   )

#------------------------------------------------------------
#introduced an Improvement of the seventSeesWithSeed
seventhSeedsTk = cms.EDProducer("ConversionSeedFilter",
                                seedCollectionPos = cms.InputTag("seventhSeedsPositive"),
                                seedCollectionNeg = cms.InputTag("seventhSeedsNegative"),
                                inputTrajectory   = cms.InputTag("generalTracksStd"),
                                deltaPhiCut = cms.double(1.5),
                                deltaCotThetaCut = cms.double(0.25),
                                deltaRCut = cms.double(5.),
                                deltaZCut = cms.double(5.),
                                maxInputSeeds = cms.uint32(200),
                                takeAll   = cms.bool(False)
                                )
# to revert to the previous uncomment this line and comment the next
#seventhSeeds = seventhSeedsWithSeeds.clone()
seventhSeeds = seventhSeedsTk.clone()

#------------------------------------------------------------



#seventhSeeds =  cms.EDProducer("ConversionSeedFilterFwk",
#                               src = cms.InputTag("seventhPLSeeds"),
#seventhSeeds =  cms.EDProducer("ConversionSeedFilter",
#                               seedCollection = cms.InputTag("seventhPLSeeds"),
#                               chargeCut = cms.bool(True),
#                               deltaPhiCut = cms.double(1.5),
#                               deltaCotThetaCut = cms.double(0.25),
#                               deltaRCut = cms.double(5.),
#                               deltaZCut = cms.double(5.)
#                               )

# TRACKER DATA CONTROL
seventhMeasurementTracker = RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi.MeasurementTracker.clone(
    ComponentName = 'seventhMeasurementTracker',
    pixelClusterProducer = 'seventhClusters',
    stripClusterProducer = 'seventhClusters'
    )
# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi
seventhCkfTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone(
    ComponentName = 'seventhCkfTrajectoryFilter',
    filterPset = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.filterPset.clone(
    maxLostHits = 1,
    minimumNumberOfHits = 3,
    minPt = 0.1
    )
    )
# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
seventhCkfTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone(
    ComponentName = 'seventhCkfTrajectoryBuilder',
    MeasurementTrackerName = 'seventhMeasurementTracker',
    trajectoryFilterName = 'seventhCkfTrajectoryFilter',
    minNrOfHitsForRebuild = 3
    )
# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
seventhTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('seventhSeeds'),
    TrajectoryBuilder = 'seventhCkfTrajectoryBuilder'
)
# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
seventhStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'seventhTrackCandidates',
    clusterRemovalInfo = 'seventhClusters',
    AlgorithmName = cms.string('iter7')
    )
# TRACK SELECTION AND QUALITY FLAG SETTING.
seventhStepLoose = RecoTracker.FinalTrackSelectors.selectLoose_cfi.selectLoose.clone(
    src = 'seventhStepTracks',
    keepAllTracks = True,
    copyExtras = False,
    copyTrajectories = True,
    applyAdaptedPVCuts = False,
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
seventhStepTight = RecoTracker.FinalTrackSelectors.selectTight_cfi.selectTight.clone(
    src = 'seventhStepLoose',
    keepAllTracks = True,
    copyExtras = False,
    copyTrajectories = True,
    applyAdaptedPVCuts = False,
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
seventhStep = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone(
    src = 'seventhStepTight',
    keepAllTracks = True,
    copyExtras = False,
    copyTrajectories = True,
    applyAdaptedPVCuts = False,
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


