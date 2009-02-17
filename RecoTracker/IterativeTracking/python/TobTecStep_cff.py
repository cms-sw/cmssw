import FWCore.ParameterSet.Config as cms

#
# Very large impact parameter tracking using TOB + TEC ring 5 seeding
#


#HIT REMOVAL
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
)


#TRACKER HITS
import RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi
import RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi
fifthPixelRecHits = RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi.siPixelRecHits.clone()
fifthStripRecHits = RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi.siStripMatchedRecHits.clone()
fifthPixelRecHits.src = 'fifthClusters'
fifthStripRecHits.ClusterProducer = 'fifthClusters'

#SEEDING LAYERS
fifthlayerpairs = cms.ESProducer("TobTecLayerPairsESProducer",
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

#SEEDS
import RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cff
fifthSeeds = RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cff.globalMixedSeeds.clone()
fifthSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'fifthlayerPairs'
fifthSeeds.RegionFactoryPSet.RegionPSet.ptMin = 0.9
fifthSeeds.RegionFactoryPSet.RegionPSet.originHalfLength = 10.0
fifthSeeds.RegionFactoryPSet.RegionPSet.originRadius = 5.0

#TRAJECTORY MEASUREMENT
import RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi
fifthMeasurementTracker = RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi.MeasurementTracker.clone()
fifthMeasurementTracker.ComponentName = 'fifthMeasurementTracker'
fifthMeasurementTracker.pixelClusterProducer = 'fifthClusters'
fifthMeasurementTracker.stripClusterProducer = 'fifthClusters'

#TRAJECTORY FILTERS (for inwards and outwards track building steps)
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi

fifthCkfTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone()
fifthCkfTrajectoryFilter.ComponentName = 'fifthCkfTrajectoryFilter'
fifthCkfTrajectoryFilter.filterPset.maxLostHits = 0
fifthCkfTrajectoryFilter.filterPset.minimumNumberOfHits = 6
fifthCkfTrajectoryFilter.filterPset.minPt = 0.6
fifthCkfTrajectoryFilter.filterPset.minHitsMinPt = 3

fifthCkfInOutTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone()
fifthCkfInOutTrajectoryFilter.ComponentName = 'fifthCkfInOutTrajectoryFilter'
fifthCkfInOutTrajectoryFilter.filterPset.maxLostHits = 0
fifthCkfInOutTrajectoryFilter.filterPset.minimumNumberOfHits = 4
fifthCkfInOutTrajectoryFilter.filterPset.minPt = 0.6
fifthCkfInOutTrajectoryFilter.filterPset.minHitsMinPt = 3

#TRAJECTORY BUILDER
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
fifthCkfTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone()
fifthCkfTrajectoryBuilder.ComponentName = 'fifthCkfTrajectoryBuilder'
fifthCkfTrajectoryBuilder.MeasurementTrackerName = 'fifthMeasurementTracker'
fifthCkfTrajectoryBuilder.trajectoryFilterName = 'fifthCkfTrajectoryFilter'
fifthCkfTrajectoryBuilder.inOutTrajectoryFilterName = 'fifthCkfInOutTrajectoryFilter'
fifthCkfTrajectoryBuilder.useSameTrajFilter = False
fifthCkfTrajectoryBuilder.minNrOfHitsForRebuild = 4
fifthCkfTrajectoryBuilder.alwaysUseInvalidHits = False
#fifthCkfTrajectoryBuilder.startSeedHitsInRebuild = True

#TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
fifthTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone()
fifthTrackCandidates.src = cms.InputTag('fifthSeeds')
fifthTrackCandidates.TrajectoryBuilder = 'fifthCkfTrajectoryBuilder'
fifthTrackCandidates.doSeedingRegionRebuilding = True
fifthTrackCandidates.useHitsSplitting = True
fifthTrackCandidates.cleanTrajectoryAfterInOut = True

#
# TRACK FITTING AND SMOOTHING
#

import TrackingTools.TrackFitters.KFFittingSmootherWithOutliersRejectionAndRK_cfi
fifthFittingSmootherWithOutlierRejection = TrackingTools.TrackFitters.KFFittingSmootherWithOutliersRejectionAndRK_cfi.KFFittingSmootherWithOutliersRejectionAndRK.clone()
fifthFittingSmootherWithOutlierRejection.ComponentName = 'fifthFittingSmootherWithOutlierRejection'
fifthFittingSmootherWithOutlierRejection.EstimateCut = 20
fifthFittingSmootherWithOutlierRejection.MinNumberOfHits = 6
fifthFittingSmootherWithOutlierRejection.Fitter = cms.string('fifthRKFitter')
fifthFittingSmootherWithOutlierRejection.Smoother = cms.string('fifthRKSmoother')

# Also necessary to specify minimum number of hits after final track fit
import TrackingTools.TrackFitters.RungeKuttaKFTrajectoryFitterESProducer_cfi
import TrackingTools.TrackFitters.RungeKuttaKFTrajectorySmootherESProducer_cfi
fifthRKTrajectoryFitter = TrackingTools.TrackFitters.RungeKuttaKFTrajectoryFitterESProducer_cfi.RKTrajectoryFitter.clone()
fifthRKTrajectorySmoother = TrackingTools.TrackFitters.RungeKuttaKFTrajectorySmootherESProducer_cfi.RKTrajectorySmoother.clone()
fifthRKTrajectoryFitter.ComponentName = cms.string('fifthRKFitter')
fifthRKTrajectorySmoother.ComponentName = cms.string('fifthRKSmoother')
fifthRKTrajectoryFitter.minHits = 6
fifthRKTrajectorySmoother.minHits = 6

#TRACKS
import RecoTracker.TrackProducer.TrackProducer_cfi
fifthWithMaterialTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone()
fifthWithMaterialTracks.src = 'fifthTrackCandidates'
fifthWithMaterialTracks.clusterRemovalInfo = 'fifthClusters'
fifthWithMaterialTracks.AlgorithmName = cms.string('iter5')
fifthWithMaterialTracks.Fitter = 'fifthFittingSmootherWithOutlierRejection'

# track selection
import RecoTracker.FinalTrackSelectors.selectLoose_cfi
import RecoTracker.FinalTrackSelectors.selectTight_cfi
import RecoTracker.FinalTrackSelectors.selectHighPurity_cfi
import RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi

tobtecStepLoose = RecoTracker.FinalTrackSelectors.selectLoose_cfi.selectLoose.clone()
tobtecStepLoose.src = 'fifthWithMaterialTracks'
tobtecStepLoose.keepAllTracks = False
tobtecStepLoose.copyExtras = True
tobtecStepLoose.copyTrajectories = True
tobtecStepLoose.chi2n_par = 0.6
tobtecStepLoose.res_par = ( 0.003, 0.001 )
tobtecStepLoose.minNumberLayers = 6
tobtecStepLoose.d0_par1 = ( 1.8, 4.0 )
tobtecStepLoose.dz_par1 = ( 1.5, 4.0 )
tobtecStepLoose.d0_par2 = ( 1.8, 4.0 )
tobtecStepLoose.dz_par2 = ( 1.5, 4.0 )

tobtecStepTight = RecoTracker.FinalTrackSelectors.selectTight_cfi.selectTight.clone()
tobtecStepTight.src = 'tobtecStepLoose'
tobtecStepTight.keepAllTracks = True
tobtecStepTight.copyExtras = True
tobtecStepTight.copyTrajectories = True
tobtecStepTight.chi2n_par = 0.35
tobtecStepTight.res_par = ( 0.003, 0.001 )
tobtecStepTight.minNumberLayers = 6
tobtecStepTight.d0_par1 = ( 1.3, 4.0 )
tobtecStepTight.dz_par1 = ( 1.2, 4.0 )
tobtecStepTight.d0_par2 = ( 1.3, 4.0 )
tobtecStepTight.dz_par2 = ( 1.2, 4.0 )

tobtecStep = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone()
tobtecStep.src = 'tobtecStepTight'
tobtecStep.keepAllTracks = True
tobtecStep.copyExtras = True
tobtecStep.copyTrajectories = True
tobtecStep.chi2n_par = 0.25
tobtecStep.res_par = ( 0.003, 0.001 )
tobtecStep.minNumberLayers = 6
tobtecStep.d0_par1 = ( 1.2, 4.0 )
tobtecStep.dz_par1 = ( 1.1, 4.0 )
tobtecStep.d0_par2 = ( 1.2, 4.0 )
tobtecStep.dz_par2 = ( 1.1, 4.0 )

fifthStep = cms.Sequence(fourthfilter*fifthClusters*
                          fifthPixelRecHits*fifthStripRecHits*
                          fifthSeeds*
                          fifthTrackCandidates*
                          fifthWithMaterialTracks*
                          tobtecStepLoose*
                          tobtecStepTight*
                          tobtecStep)
