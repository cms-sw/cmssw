import FWCore.ParameterSet.Config as cms

#######################################################################
# Very large impact parameter tracking using TOB + TEC ring 5 seeding #
#######################################################################

# REMOVE HITS ASSIGNED TO GOOD TRACKS FROM PREVIOUS ITERATIONS

fourthfilter = cms.EDProducer("QualityFilter",
    TrackQuality = cms.string('highPurity'),
    recTracks = cms.InputTag("pixellessStep")
)

fifthClusters = cms.EDProducer("TrackClusterRemover",
    oldClusterRemovalInfo = cms.InputTag("fourthClusters"),
    trajectories = cms.InputTag("fourthfilter"),
    pixelClusters = cms.InputTag("fourthClusters"),
    stripClusters = cms.InputTag("fourthClusters"),
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
fifthPixelRecHits = RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi.siPixelRecHits.clone(
    src = 'fifthClusters'
    )
fifthStripRecHits = RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi.siStripMatchedRecHits.clone(
    ClusterProducer = 'fifthClusters'
    )

# SEEDING LAYERS
fifthlayerpairs = cms.ESProducer("SeedingLayersESProducer",
    ComponentName = cms.string('fifthlayerPairs'),

    layerList = cms.vstring('TOB1+TOB2', 
        'TOB1+TEC1_pos', 'TOB1+TEC1_neg', 
        'TEC1_pos+TEC2_pos', 'TEC2_pos+TEC3_pos', 
        'TEC3_pos+TEC4_pos', 'TEC4_pos+TEC5_pos', 
        'TEC5_pos+TEC6_pos', 'TEC6_pos+TEC7_pos', 
        'TEC1_neg+TEC2_neg', 'TEC2_neg+TEC3_neg', 
        'TEC3_neg+TEC4_neg', 'TEC4_neg+TEC5_neg', 
        'TEC5_neg+TEC6_neg', 'TEC6_neg+TEC7_neg'),

    TOB = cms.PSet(
        matchedRecHits = cms.InputTag("fifthStripRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),

    TEC = cms.PSet(
        matchedRecHits = cms.InputTag("fifthStripRecHits","matchedRecHit"),
        #    untracked bool useSimpleRphiHitsCleaner = false
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        minRing = cms.int32(5),
        maxRing = cms.int32(5)
    )
)
# SEEDS
import RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cff
fifthSeeds = RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cff.globalMixedSeeds.clone()
fifthSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'fifthlayerPairs'
fifthSeeds.RegionFactoryPSet.RegionPSet.ptMin = 0.6
fifthSeeds.RegionFactoryPSet.RegionPSet.originHalfLength = 30.0
fifthSeeds.RegionFactoryPSet.RegionPSet.originRadius = 6.0
fifthSeeds.ClusterCheckPSet.PixelClusterCollectionLabel = 'fifthClusters'
fifthSeeds.ClusterCheckPSet.ClusterCollectionLabel = 'fifthClusters'
   

# TRACKER DATA CONTROL
import RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi
fifthMeasurementTracker = RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi.MeasurementTracker.clone(
    ComponentName = 'fifthMeasurementTracker',
    pixelClusterProducer = 'fifthClusters',
    stripClusterProducer = 'fifthClusters'
    )

# QUALITY CUTS DURING TRACK BUILDING (for inwardss and outwards track building steps)
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi

fifthCkfTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone(
    ComponentName = 'fifthCkfTrajectoryFilter',
    filterPset = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.filterPset.clone(
    maxLostHits = 0,
    minimumNumberOfHits = 6,
    minPt = 0.1,
    minHitsMinPt = 3
    )
    )

fifthCkfInOutTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone(
    ComponentName = 'fifthCkfInOutTrajectoryFilter',
    filterPset = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.filterPset.clone(
    maxLostHits = 0,
    minimumNumberOfHits = 4,
    minPt = 0.1,
    minHitsMinPt = 3
    )
    )

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
fifthCkfTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone(
    ComponentName = 'fifthCkfTrajectoryBuilder',
    MeasurementTrackerName = 'fifthMeasurementTracker',
    trajectoryFilterName = 'fifthCkfTrajectoryFilter',
    inOutTrajectoryFilterName = 'fifthCkfInOutTrajectoryFilter',
    useSameTrajFilter = False,
    minNrOfHitsForRebuild = 4,
    alwaysUseInvalidHits = False
    #startSeedHitsInRebuild = True
    )

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
fifthTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('fifthSeeds'),
    TrajectoryBuilder = 'fifthCkfTrajectoryBuilder',
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True,
    cleanTrajectoryAfterInOut = True
    )

# TRACK FITTING AND SMOOTHING OPTIONS
import TrackingTools.TrackFitters.RungeKuttaFitters_cff
fifthFittingSmootherWithOutlierRejection = TrackingTools.TrackFitters.RungeKuttaFitters_cff.KFFittingSmootherWithOutliersRejectionAndRK.clone(
    ComponentName = 'fifthFittingSmootherWithOutlierRejection',
    EstimateCut = 30,
    MinNumberOfHits = 8,
    Fitter = cms.string('fifthRKFitter'),
    Smoother = cms.string('fifthRKSmoother')
    )

# Also necessary to specify minimum number of hits after final track fit
fifthRKTrajectoryFitter = TrackingTools.TrackFitters.RungeKuttaFitters_cff.RKTrajectoryFitter.clone(
    ComponentName = cms.string('fifthRKFitter'),
    minHits = 8
    )

fifthRKTrajectorySmoother = TrackingTools.TrackFitters.RungeKuttaFitters_cff.RKTrajectorySmoother.clone(
    ComponentName = cms.string('fifthRKSmoother'),
    errorRescaling = 10.0,
    minHits = 8
    )

# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
fifthWithMaterialTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'fifthTrackCandidates',
    clusterRemovalInfo = 'fifthClusters',
    AlgorithmName = cms.string('iter5'),
    Fitter = 'fifthFittingSmootherWithOutlierRejection',
    )


import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
tobtecSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src='fifthWithMaterialTracks',
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'tobtecStepLoose',
            chi2n_par = 0.4,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 5,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 2,
            d0_par1 = ( 2.0, 4.0 ),
            dz_par1 = ( 1.8, 4.0 ),
            d0_par2 = ( 2.0, 4.0 ),
            dz_par2 = ( 1.8, 4.0 )
            ), #end of pset for thStepVtxLoose
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'tobtecStepTight',
            preFilterName = 'tobtecStepLoose',
            chi2n_par = 0.3,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 5,
            maxNumberLostLayers = 0,
            minNumber3DLayers = 2,
            d0_par1 = ( 1.5, 4.0 ),
            dz_par1 = ( 1.4, 4.0 ),
            d0_par2 = ( 1.5, 4.0 ),
            dz_par2 = ( 1.4, 4.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'tobtecStep',
            preFilterName = 'tobtecStepTight',
            chi2n_par = 0.2,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 5,
            maxNumberLostLayers = 0,
            minNumber3DLayers = 2,
            d0_par1 = ( 1.4, 4.0 ),
            dz_par1 = ( 1.3, 4.0 ),
            d0_par2 = ( 1.4, 4.0 ),
            dz_par2 = ( 1.3, 4.0 )
            ),
        ) #end of vpset
    ) #end of clone


import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
tobtecStep = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = cms.VInputTag(cms.InputTag('fifthWithMaterialTracks')),
    hasSelector=cms.vint32(1),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("tobtecSelector","tobtecStep")),
    setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0), pQual=cms.bool(True) ))
)                        


fifthStep = cms.Sequence(fourthfilter*fifthClusters*
                          fifthPixelRecHits*fifthStripRecHits*
                          fifthSeeds*
                          fifthTrackCandidates*
                          fifthWithMaterialTracks*
                          tobtecSelector*
                          tobtecStep)
