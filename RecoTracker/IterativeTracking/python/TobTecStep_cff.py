import FWCore.ParameterSet.Config as cms

#######################################################################
# Very large impact parameter tracking using TOB + TEC ring 5 seeding #
#######################################################################

# REMOVE HITS ASSIGNED TO GOOD TRACKS FROM PREVIOUS ITERATIONS

fourthfilter = cms.EDFilter("QualityFilter",
    TrackQuality = cms.string('highPurity'),
    recTracks = cms.InputTag("pixellessStep")
)

fifthClusters = cms.EDFilter("TrackClusterRemover",
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
    ComponentName = cms.string('TobTecLayerPairs'),

    layerList = cms.vstring('TOB1+TOB2', 
        'TOB1+TEC1_pos', 'TOB1+TEC1_neg', 
        'TEC1_pos+TEC2_pos', 'TEC2_pos+TEC3_pos', 
        'TEC3_pos+TEC4_pos', 'TEC4_pos+TEC5_pos', 
        'TEC5_pos+TEC6_pos', 'TEC6_pos+TEC7_pos', 
        'TEC1_neg+TEC2_neg', 'TEC2_neg+TEC3_neg', 
        'TEC3_neg+TEC4_neg', 'TEC4_neg+TEC5_neg', 
        'TEC5_neg+TEC6_neg', 'TEC6_neg+TEC7_neg'),

    TOB = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),

    TEC = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        #    untracked bool useSimpleRphiHitsCleaner = false
        useRingSlector = cms.untracked.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        minRing = cms.int32(5),
        maxRing = cms.int32(5)
    )
)
fifthlayerpairs.ComponentName = 'fifthlayerPairs'
fifthlayerpairs.TOB.matchedRecHits = 'fifthStripRecHits:matchedRecHit'
fifthlayerpairs.TEC.matchedRecHits = 'fifthStripRecHits:matchedRecHit'

# SEEDS
import RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cff
fifthSeeds = RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cff.globalMixedSeeds.clone()
fifthSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'fifthlayerPairs'
fifthSeeds.RegionFactoryPSet.RegionPSet.ptMin = 0.8
fifthSeeds.RegionFactoryPSet.RegionPSet.originHalfLength = 10.0
fifthSeeds.RegionFactoryPSet.RegionPSet.originRadius = 5.0

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
    minimumNumberOfHits = 7,
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
    alwaysUseInvalidHits = False,
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
import TrackingTools.TrackFitters.KFFittingSmootherWithOutliersRejectionAndRK_cfi
fifthFittingSmootherWithOutlierRejection = TrackingTools.TrackFitters.KFFittingSmootherWithOutliersRejectionAndRK_cfi.KFFittingSmootherWithOutliersRejectionAndRK.clone(
    ComponentName = 'fifthFittingSmootherWithOutlierRejection',
    EstimateCut = 20,
    MinNumberOfHits = 7,
    Fitter = cms.string('fifthRKFitter'),
    Smoother = cms.string('fifthRKSmoother')
    )

# Also necessary to specify minimum number of hits after final track fit
import TrackingTools.TrackFitters.RungeKuttaKFTrajectoryFitterESProducer_cfi
import TrackingTools.TrackFitters.RungeKuttaKFTrajectorySmootherESProducer_cfi
fifthRKTrajectoryFitter = TrackingTools.TrackFitters.RungeKuttaKFTrajectoryFitterESProducer_cfi.RKTrajectoryFitter.clone(
    ComponentName = cms.string('fifthRKFitter'),
    minHits = 7
    )

fifthRKTrajectorySmoother = TrackingTools.TrackFitters.RungeKuttaKFTrajectorySmootherESProducer_cfi.RKTrajectorySmoother.clone(
    ComponentName = cms.string('fifthRKSmoother'),
    minHits = 7
    )

# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
fifthWithMaterialTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'fifthTrackCandidates',
    clusterRemovalInfo = 'fifthClusters',
    AlgorithmName = cms.string('iter5'),
    Fitter = 'fifthFittingSmootherWithOutlierRejection',
    )

# TRACK SELECTION AND QUALITY FLAG SETTING.
import RecoTracker.FinalTrackSelectors.selectLoose_cfi
import RecoTracker.FinalTrackSelectors.selectTight_cfi
import RecoTracker.FinalTrackSelectors.selectHighPurity_cfi
import RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi

tobtecStepLoose = RecoTracker.FinalTrackSelectors.selectLoose_cfi.selectLoose.clone(
    src = 'fifthWithMaterialTracks',
    keepAllTracks = False,
    copyExtras = True,
    copyTrajectories = True,
    chi2n_par = 0.6,
    res_par = ( 0.003, 0.001 ),
    minNumberLayers = 6,
    maxNumberLostLayers = 1,
    minNumber3DLayers = 2,
    d0_par1 = ( 1.8, 4.0 ),
    dz_par1 = ( 1.5, 4.0 ),
    d0_par2 = ( 1.8, 4.0 ),
    dz_par2 = ( 1.5, 4.0 )
    )

tobtecStepTight = RecoTracker.FinalTrackSelectors.selectTight_cfi.selectTight.clone(
    src = 'tobtecStepLoose',
    keepAllTracks = True,
    copyExtras = True,
    copyTrajectories = True,
    chi2n_par = 0.35,
    res_par = ( 0.003, 0.001 ),
    minNumberLayers = 6,
    maxNumberLostLayers = 0,
    minNumber3DLayers = 2,
    d0_par1 = ( 1.3, 4.0 ),
    dz_par1 = ( 1.2, 4.0 ),
    d0_par2 = ( 1.3, 4.0 ),
    dz_par2 = ( 1.2, 4.0 )
    )

tobtecStep = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone(
    src = 'tobtecStepTight',
    keepAllTracks = True,
    copyExtras = True,
    copyTrajectories = True,
    chi2n_par = 0.25,
    res_par = ( 0.003, 0.001 ),
    minNumberLayers = 6,
    maxNumberLostLayers = 0,
    minNumber3DLayers = 2,
    d0_par1 = ( 1.2, 4.0 ),
    dz_par1 = ( 1.1, 4.0 ),
    d0_par2 = ( 1.2, 4.0 ),
    dz_par2 = ( 1.1, 4.0 )
    )

fifthStep = cms.Sequence(fourthfilter*fifthClusters*
                          fifthPixelRecHits*fifthStripRecHits*
                          fifthSeeds*
                          fifthTrackCandidates*
                          fifthWithMaterialTracks*
                          tobtecStepLoose*
                          tobtecStepTight*
                          tobtecStep)
