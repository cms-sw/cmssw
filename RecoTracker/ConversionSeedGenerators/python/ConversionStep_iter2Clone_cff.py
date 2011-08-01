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

# SEEDS
import RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff
from RecoTracker.TkTrackingRegions.GlobalTrackingRegionFromBeamSpot_cfi import RegionPsetFomBeamSpotBlock

###########################################################################################
# SEEDING LAYERS
sixthlayertripletsA = cms.ESProducer("SeedingLayersESProducer",
                                     ComponentName = cms.string('SixthLayerTripletsA'),
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


sixthTripletsA = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone(
    RegionFactoryPSet = RegionPsetFomBeamSpotBlock.clone(
        ComponentName = cms.string('GlobalRegionProducerFromBeamSpot'),
        RegionPSet = RegionPsetFomBeamSpotBlock.RegionPSet.clone(
            originRadius = cms.double(6),
            ptMin = 0.075,
            nSigmaZ = 3.3
            )
        )
    )

sixthTripletsA.OrderedHitsFactoryPSet.SeedingLayers = 'SixthLayerTripletsA'
sixthTripletsA.SeedCreatorPSet.ComponentName = 'SeedFromConsecutiveHitsTripletOnlyCreator'
#sixthTriplets.OrderedHitsFactoryPSet.GeneratorPSet = cms.PSet(PixelTripletLargeTipGenerator)

sixthlayertripletsB = cms.ESProducer("SeedingLayersESProducer",
                                     ComponentName = cms.string('SixthLayerTripletsB'),
                                        layerList = cms.vstring('BPix1+BPix2+BPix3', 
                                                                'BPix1+BPix2+FPix1_pos', 'BPix1+BPix2+FPix1_neg', 
                                                                'BPix3+FPix1_pos+TID1_pos', 'BPix3+FPix1_neg+TID1_neg', 
                                                                'FPix1_pos+FPix2_pos+TEC1_pos', 'FPix1_neg+FPix2_neg+TEC1_neg',
                                                                'FPix2_pos+TID3_pos+TEC1_pos', 'FPix2_neg+TID3_neg+TEC1_neg',
                                                                'FPix2_pos+TEC2_pos+TEC3_pos', 'FPix2_neg+TEC2_neg+TEC3_neg'),
                                     TEC = cms.PSet(
                                         matchedRecHits = cms.InputTag("sixthStripRecHits","matchedRecHit"),
                                         useRingSlector = cms.bool(True),
                                         TTRHBuilder = cms.string('WithTrackAngle'),
                                         minRing = cms.int32(1),
                                         maxRing = cms.int32(2)
                                         ),
                                     FPix = cms.PSet(
                                         useErrorsFromParam = cms.bool(True),
                                         hitErrorRPhi = cms.double(0.0051),
                                         TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedTriplets'),
                                         HitProducer = cms.string('sixthPixelRecHits'),
                                         hitErrorRZ = cms.double(0.0036)
                                         ),
                                     TID = cms.PSet(
                                         matchedRecHits = cms.InputTag("sixthStripRecHits","matchedRecHit"),
                                         useRingSlector = cms.bool(True),
                                         TTRHBuilder = cms.string('WithTrackAngle'),
                                         minRing = cms.int32(1),
                                         maxRing = cms.int32(2)
                                         ),
                                     BPix = cms.PSet(
                                         useErrorsFromParam = cms.bool(True),
                                         hitErrorRPhi = cms.double(0.0027),
                                         TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedTriplets'),
                                         HitProducer = cms.string('sixthPixelRecHits'),
                                         hitErrorRZ = cms.double(0.006)
                                         )
                                     )


sixthTripletsB = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone(
    RegionFactoryPSet = RegionPsetFomBeamSpotBlock.clone(
        ComponentName = cms.string('GlobalRegionProducerFromBeamSpot'),
        RegionPSet = RegionPsetFomBeamSpotBlock.RegionPSet.clone(
            originRadius = cms.double(6),
            ptMin = 0.075,
            nSigmaZ = 3.3
            )
        )
    )

sixthTripletsB.OrderedHitsFactoryPSet.SeedingLayers = 'SixthLayerTripletsB'
sixthTripletsB.SeedCreatorPSet.ComponentName = 'SeedFromConsecutiveHitsTripletOnlyCreator'
#sixthTriplets.OrderedHitsFactoryPSet.GeneratorPSet = cms.PSet(PixelTripletLargeTipGenerator)


from  RecoTracker.TkSeedGenerator.GlobalCombinedSeeds_cfi import *
sixthTriplets = globalCombinedSeeds.clone(seedCollections = cms.VInputTag(cms.InputTag('sixthTripletsA'),
                                                                          cms.InputTag('sixthTripletsB'),
                                                                          )                                         
                                          )                                          
                                          ##################################################################################################################

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
    minPt = 0.075
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
    src = cms.InputTag('sixthTriplets'),
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
