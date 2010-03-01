import FWCore.ParameterSet.Config as cms

############################################################
# Large impact parameter Tracking using mixed-pair seeding #
############################################################

# REMOVE HITS ASSIGNED TO GOOD TRACKS FROM PREVIOUS ITERATIONS
secfilter = cms.EDProducer("QualityFilter",
    TrackQuality = cms.string('highPurity'),
    recTracks = cms.InputTag("secStep")
)

thClusters = cms.EDProducer("TrackClusterRemover",
    oldClusterRemovalInfo = cms.InputTag("secClusters"),
    trajectories = cms.InputTag("secfilter"),
    pixelClusters = cms.InputTag("secClusters"),
    stripClusters = cms.InputTag("secClusters"),
    Common = cms.PSet(
        maxChi2 = cms.double(30.0)
    )

# For debug purposes, you can run this iteration not eliminating any hits from previous ones by
# instead using
#    trajectories = cms.InputTag("zeroStepFilter"),
#    pixelClusters = cms.InputTag("siPixelClusters"),
#    stripClusters = cms.InputTag("siStripClusters"),
#     Common = cms.PSet(
#       maxChi2 = cms.double(0.0)
#    )
)

# TRACKER HITS
import RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi
thPixelRecHits = RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi.siPixelRecHits.clone(
    src = 'thClusters'
    )
import RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi
thStripRecHits = RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi.siStripMatchedRecHits.clone(
    ClusterProducer = 'thClusters'
    )

# Propagator taking into account momentum uncertainty in multiple scattering calculation.

import TrackingTools.MaterialEffects.MaterialPropagator_cfi
MaterialPropagatorPtMin01 = TrackingTools.MaterialEffects.MaterialPropagator_cfi.MaterialPropagator.clone(
    ComponentName = 'PropagatorWithMaterialPtMin01',
    ptMin = 0.1
    )

import TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi
OppositeMaterialPropagatorPtMin01 = TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi.OppositeMaterialPropagator.clone(
    ComponentName = 'PropagatorWithMaterialOppositePtMin01',
    ptMin = 0.1
    )

# SEEDING LAYERS
thlayertripletsa = cms.ESProducer("SeedingLayersESProducer",
    ComponentName = cms.string('ThLayerTripletsA'),
    layerList = cms.vstring('BPix1+BPix2+BPix3', 
        'BPix1+BPix2+FPix1_pos', 'BPix1+BPix2+FPix1_neg', 
        'BPix3+FPix1_pos+TID1_pos', 'BPix3+FPix1_neg+TID1_neg', 
        'FPix1_pos+FPix2_pos+TEC1_pos', 'FPix1_neg+FPix2_neg+TEC1_neg',
        'FPix2_pos+TID3_pos+TEC1_pos', 'FPix2_neg+TID3_neg+TEC1_neg',
        'FPix2_pos+TEC2_pos+TEC3_pos', 'FPix2_neg+TEC2_neg+TEC3_neg'),
    TEC = cms.PSet(
        matchedRecHits = cms.InputTag("thStripRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        minRing = cms.int32(1),
        maxRing = cms.int32(2)
    ),
    FPix = cms.PSet(
        useErrorsFromParam = cms.bool(True),
        hitErrorRPhi = cms.double(0.0051),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedTriplets'),
        HitProducer = cms.string('thPixelRecHits'),
        hitErrorRZ = cms.double(0.0036)
    ),
    TID = cms.PSet(
        matchedRecHits = cms.InputTag("thStripRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        minRing = cms.int32(1),
        maxRing = cms.int32(2)
    ),
    BPix = cms.PSet(
        useErrorsFromParam = cms.bool(True),
        hitErrorRPhi = cms.double(0.0027),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedTriplets'),
        HitProducer = cms.string('thPixelRecHits'),
        hitErrorRZ = cms.double(0.006)
    ),
    TIB = cms.PSet(
        matchedRecHits = cms.InputTag("thStripRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    )
)


# SEEDS
from RecoPixelVertexing.PixelTriplets.PixelTripletLargeTipGenerator_cfi import *
PixelTripletLargeTipGenerator.extraHitRZtolerance = 0.0
PixelTripletLargeTipGenerator.extraHitRPhitolerance = 0.0
import RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff
thTripletsA = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone()
thTripletsA.OrderedHitsFactoryPSet.SeedingLayers = 'ThLayerTripletsA'
thTripletsA.OrderedHitsFactoryPSet.GeneratorPSet = cms.PSet(PixelTripletLargeTipGenerator)
thTripletsA.SeedCreatorPSet.ComponentName = 'SeedFromConsecutiveHitsTripletOnlyCreator'
thTripletsA.RegionFactoryPSet.RegionPSet.ptMin = 0.25
thTripletsA.RegionFactoryPSet.RegionPSet.originHalfLength = 10.0
thTripletsA.RegionFactoryPSet.RegionPSet.originRadius = 2.0


thlayertripletsb = cms.ESProducer("SeedingLayersESProducer",
    ComponentName = cms.string('ThLayerTripletsB'),
    layerList = cms.vstring('BPix2+BPix3+TIB1', 
        'BPix2+BPix3+TIB2','BPix3+TIB1+TIB2'),
    BPix = cms.PSet(
        useErrorsFromParam = cms.bool(True),
        hitErrorRPhi = cms.double(0.0027),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedTriplets'),
        HitProducer = cms.string('thPixelRecHits'),
        hitErrorRZ = cms.double(0.006)
    ),
    TIB = cms.PSet(
        matchedRecHits = cms.InputTag("thStripRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    )
)


# SEEDS
from RecoPixelVertexing.PixelTriplets.PixelTripletLargeTipGenerator_cfi import *
PixelTripletLargeTipGenerator.extraHitRZtolerance = 0.0
PixelTripletLargeTipGenerator.extraHitRPhitolerance = 0.0
import RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff
thTripletsB = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone()
thTripletsB.OrderedHitsFactoryPSet.SeedingLayers = 'ThLayerTripletsB'
thTripletsB.OrderedHitsFactoryPSet.GeneratorPSet = cms.PSet(PixelTripletLargeTipGenerator)
thTripletsB.SeedCreatorPSet.ComponentName = 'SeedFromConsecutiveHitsTripletOnlyCreator'
thTripletsB.RegionFactoryPSet.RegionPSet.ptMin = 0.35
thTripletsB.RegionFactoryPSet.RegionPSet.originHalfLength = 10.0
thTripletsB.RegionFactoryPSet.RegionPSet.originRadius = 2.0


import RecoTracker.TkSeedGenerator.GlobalCombinedSeeds_cfi
thTriplets = RecoTracker.TkSeedGenerator.GlobalCombinedSeeds_cfi.globalCombinedSeeds.clone()
thTriplets.seedCollections = cms.VInputTag(
        cms.InputTag('thTripletsA'),
        cms.InputTag('thTripletsB'),
        )


# TRACKER DATA CONTROL
import RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi
thMeasurementTracker = RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi.MeasurementTracker.clone(
    ComponentName = 'thMeasurementTracker',
    pixelClusterProducer = 'thClusters',
    stripClusterProducer = 'thClusters'
    )

# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi
thCkfTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone(
    ComponentName = 'thCkfTrajectoryFilter',
    filterPset = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.filterPset.clone(
    maxLostHits = 0,
    minimumNumberOfHits = 3,
    minPt = 0.1
    )
    )

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
thCkfTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone(
    ComponentName = 'thCkfTrajectoryBuilder',
    MeasurementTrackerName = 'thMeasurementTracker',
    trajectoryFilterName = 'thCkfTrajectoryFilter',
    propagatorAlong = cms.string('PropagatorWithMaterialPtMin01'),
    propagatorOpposite = cms.string('PropagatorWithMaterialOppositePtMin01')
    )

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
thTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('thTriplets'),
    TrajectoryBuilder = 'thCkfTrajectoryBuilder',
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True
)
# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
thWithMaterialTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    AlgorithmName = cms.string('iter3'),
    src = 'thTrackCandidates',
    clusterRemovalInfo = 'thClusters',
)

# TRACK SELECTION AND QUALITY FLAG SETTING.
import RecoTracker.FinalTrackSelectors.selectLoose_cfi
import RecoTracker.FinalTrackSelectors.selectTight_cfi
import RecoTracker.FinalTrackSelectors.selectHighPurity_cfi
import RecoTracker.FinalTrackSelectors.simpleTrackListMerger_cfi

thStepVtxLoose = RecoTracker.FinalTrackSelectors.selectLoose_cfi.selectLoose.clone(
    src = 'thWithMaterialTracks',
    keepAllTracks = False,
    copyExtras = False,
    copyTrajectories = True,
    chi2n_par = 1.2,
    res_par = ( 0.003, 0.001 ),
    minNumberLayers = 3,
    maxNumberLostLayers = 1,
    minNumber3DLayers = 2,
    d0_par1 = ( 1.2, 3.0 ),
    dz_par1 = ( 1.2, 3.0 ),
    d0_par2 = ( 1.3, 3.0 ),
    dz_par2 = ( 1.3, 3.0 )
    )

thStepTrkLoose = RecoTracker.FinalTrackSelectors.selectLoose_cfi.selectLoose.clone(
    src = 'thWithMaterialTracks',
    keepAllTracks = False,
    copyExtras = False,
    copyTrajectories = True,
    chi2n_par = 0.8,
    res_par = ( 0.003, 0.001 ),
    minNumberLayers = 4,
    maxNumberLostLayers = 1,
    minNumber3DLayers = 2,
    d0_par1 = ( 1.8, 4.0 ),
    dz_par1 = ( 1.8, 4.0 ),
    d0_par2 = ( 1.8, 4.0 ),
    dz_par2 = ( 1.8, 4.0 )
    )

thStepLoose = RecoTracker.FinalTrackSelectors.simpleTrackListMerger_cfi.simpleTrackListMerger.clone(
    TrackProducer1 = 'thStepVtxLoose',
    TrackProducer2 = 'thStepTrkLoose'
    )


thStepVtxTight = RecoTracker.FinalTrackSelectors.selectTight_cfi.selectTight.clone(
    src = 'thStepVtxLoose',
    keepAllTracks = True,
    copyExtras = False,
    copyTrajectories = True,
    chi2n_par = 0.7,
    res_par = ( 0.003, 0.001 ),
    minNumberLayers = 3,
    maxNumberLostLayers = 1,
    minNumber3DLayers = 3,
    d0_par1 = ( 1.0, 3.0 ),
    dz_par1 = ( 1.0, 3.0 ),
    d0_par2 = ( 1.1, 3.0 ),
    dz_par2 = ( 1.1, 3.0 )
    )

thStepTrkTight = RecoTracker.FinalTrackSelectors.selectTight_cfi.selectTight.clone(
    src = 'thStepTrkLoose',
    keepAllTracks = True,
    copyExtras = False,
    copyTrajectories = True,
    chi2n_par = 0.6,
    res_par = ( 0.003, 0.001 ),
    minNumberLayers = 5,
    maxNumberLostLayers = 1,
    minNumber3DLayers = 3,
    d0_par1 = ( 1.1, 4.0 ),
    dz_par1 = ( 1.1, 4.0 ),
    d0_par2 = ( 1.1, 4.0 ),
    dz_par2 = ( 1.1, 4.0 )
)

thStepTight = RecoTracker.FinalTrackSelectors.simpleTrackListMerger_cfi.simpleTrackListMerger.clone(
    TrackProducer1 = 'thStepVtxTight',
    TrackProducer2 = 'thStepTrkTight'
    )


thStepVtx = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone(
    src = 'thStepVtxTight',
    keepAllTracks = True,
    copyExtras = False,
    copyTrajectories = True,
    chi2n_par = 0.6,
    res_par = ( 0.003, 0.001 ),
    minNumberLayers = 3,
    maxNumberLostLayers = 1,
    minNumber3DLayers = 3,
    d0_par1 = ( 0.9, 3.0 ),
    dz_par1 = ( 0.9, 3.0 ),
    d0_par2 = ( 1.0, 3.0 ),
    dz_par2 = ( 1.0, 3.0 )
)

thStepTrk = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone(
    src = 'thStepTrkTight',
    keepAllTracks = True,
    copyExtras = False,
    copyTrajectories = True,
    chi2n_par = 0.4,
    res_par = ( 0.003, 0.001 ),
    minNumberLayers = 5,
    maxNumberLostLayers = 1,
    minNumber3DLayers = 4,
    d0_par1 = ( 1.0, 4.0 ),
    dz_par1 = ( 1.0, 4.0 ),
    d0_par2 = ( 1.0, 4.0 ),
    dz_par2 = ( 1.0, 4.0 )
    )

thStep = RecoTracker.FinalTrackSelectors.simpleTrackListMerger_cfi.simpleTrackListMerger.clone(
    TrackProducer1 = 'thStepVtx',
    TrackProducer2 = 'thStepTrk'
    )

thirdStep = cms.Sequence(secfilter*
                         thClusters*
                         thPixelRecHits*thStripRecHits*
                         thTripletsA*thTripletsB*thTriplets*
                         thTrackCandidates*
                         thWithMaterialTracks*
                         thStepVtxLoose*thStepTrkLoose*thStepLoose*
                         thStepVtxTight*thStepTrkTight*thStepTight*
                         thStepVtx*thStepTrk*thStep)
