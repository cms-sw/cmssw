import FWCore.ParameterSet.Config as cms

##########################################################################
# Large impact parameter tracking using TIB/TID/TEC stereo layer seeding #
##########################################################################

# REMOVE HITS ASSIGNED TO GOOD TRACKS FROM PREVIOUS ITERATIONS

thfilter = cms.EDProducer("QualityFilter",
    TrackQuality = cms.string('highPurity'),
    recTracks = cms.InputTag("thStep")
)

fourthClusters = cms.EDProducer("TrackClusterRemover",
    oldClusterRemovalInfo = cms.InputTag("thClusters"),
    trajectories = cms.InputTag("thfilter"),
    pixelClusters = cms.InputTag("thClusters"),
    stripClusters = cms.InputTag("thClusters"),
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
import RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi
fourthPixelRecHits = RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi.siPixelRecHits.clone(
    src = 'fourthClusters'
    )
fourthStripRecHits = RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi.siStripMatchedRecHits.clone(
    ClusterProducer = 'fourthClusters'
    )


# SEEDING LAYERS
fourthlayerpairs = cms.ESProducer("SeedingLayersESProducer",
    ComponentName = cms.string('FourthLayerPairs'),
    layerList = cms.vstring('TIB1+TIB2',
        'TIB1+TID1_pos','TIB1+TID1_neg',
        'TID1_pos+TID2_pos','TID2_pos+TID3_pos','TID3_pos+TEC1_pos',
        'TEC1_pos+TEC2_pos','TEC2_pos+TEC3_pos','TEC3_pos+TEC4_pos','TEC3_pos+TEC5_pos','TEC4_pos+TEC5_pos',
        'TID1_neg+TID2_neg','TID2_neg+TID3_neg','TID3_neg+TEC1_neg',
        'TEC1_neg+TEC2_neg','TEC2_neg+TEC3_neg','TEC3_neg+TEC4_neg','TEC3_neg+TEC5_neg','TEC4_neg+TEC5_neg'),
    TIB = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        matchedRecHits = cms.InputTag("fourthStripRecHits","matchedRecHit")
    ),
    TID = cms.PSet(
        matchedRecHits = cms.InputTag("fourthStripRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        minRing = cms.int32(1),
        maxRing = cms.int32(2)
    ),
    TEC = cms.PSet(
        matchedRecHits = cms.InputTag("fourthStripRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        minRing = cms.int32(1),
        maxRing = cms.int32(2)
    )
)

# SEEDS
import RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cff
fourthPLSeeds = RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cff.globalMixedSeeds.clone()
import RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi
fourthPLSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'FourthLayerPairs'
fourthPLSeeds.RegionFactoryPSet.RegionPSet.ptMin = 0.5
fourthPLSeeds.RegionFactoryPSet.RegionPSet.originHalfLength = 12.0
fourthPLSeeds.RegionFactoryPSet.RegionPSet.originRadius = 2.0
fourthPLSeeds.ClusterCheckPSet.PixelClusterCollectionLabel = 'fourthClusters'
fourthPLSeeds.ClusterCheckPSet.ClusterCollectionLabel = 'fourthClusters'
     
# TRACKER DATA CONTROL
fourthMeasurementTracker = RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi.MeasurementTracker.clone(
    ComponentName = 'fourthMeasurementTracker',
    pixelClusterProducer = 'fourthClusters',
    stripClusterProducer = 'fourthClusters'
    )

# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi
fourthCkfTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone(
    ComponentName = 'fourthCkfTrajectoryFilter',
    filterPset = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.filterPset.clone(
    maxLostHits = 0,
    minimumNumberOfHits = 6,
    minPt = 0.1
    )
    )

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
fourthCkfTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone(
    ComponentName = 'fourthCkfTrajectoryBuilder',
    MeasurementTrackerName = 'fourthMeasurementTracker',
    trajectoryFilterName = 'fourthCkfTrajectoryFilter',
    minNrOfHitsForRebuild = 4
    )

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
fourthTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('fourthPLSeeds'),
    TrajectoryBuilder = 'fourthCkfTrajectoryBuilder'
    )

# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
fourthWithMaterialTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'fourthTrackCandidates',
    clusterRemovalInfo = 'fourthClusters',
    AlgorithmName = cms.string('iter4')
    )


import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
pixellessSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src='fourthWithMaterialTracks',
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'pixellessStepLoose',
            chi2n_par = 0.5,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 5,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 3,
            d0_par1 = ( 1.5, 4.0 ),
            dz_par1 = ( 1.5, 4.0 ),
            d0_par2 = ( 1.5, 4.0 ),
            dz_par2 = ( 1.5, 4.0 )
            ), #end of pset for thStepVtxLoose
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'pixellessStepTight',
            preFilterName = 'pixellessStepLoose',
            chi2n_par = 0.35,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 5,
            maxNumberLostLayers = 0,
            minNumber3DLayers = 3,
            d0_par1 = ( 1.2, 4.0 ),
            dz_par1 = ( 1.2, 4.0 ),
            d0_par2 = ( 1.2, 4.0 ),
            dz_par2 = ( 1.2, 4.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'pixellessStep',
            preFilterName = 'pixellessStepTight',
            chi2n_par = 0.25,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 5,
            maxNumberLostLayers = 0,
            minNumber3DLayers = 3,
            d0_par1 = ( 1., 4.0 ),
            dz_par1 = ( 1., 4.0 ),
            d0_par2 = ( 1., 4.0 ),
            dz_par2 = ( 1., 4.0 )
            ),
        ) #end of vpset
    ) #end of clone


import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
pixellessStep = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = cms.VInputTag(cms.InputTag('fourthWithMaterialTracks')),
    hasSelector=cms.vint32(1),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("pixellessSelector","pixellessStep")),
    setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0), pQual=cms.bool(True) ))
)                        




fourthStep = cms.Sequence(thfilter*fourthClusters*
                          fourthPixelRecHits*fourthStripRecHits*
                          fourthPLSeeds*
                          fourthTrackCandidates*
                          fourthWithMaterialTracks*
                          pixellessSelector*
#                          pixellessStepLoose*
#                          pixellessStepTight*
                          pixellessStep)
