import FWCore.ParameterSet.Config as cms

#HIT REMOVAL
thfilter = cms.EDFilter("QualityFilter",
    TrackQuality = cms.string('highPurity'),
    recTracks = cms.InputTag("thStep")
)

fourthClusters = cms.EDFilter("TrackClusterRemover",
    oldClusterRemovalInfo = cms.InputTag("thClusters"),
    trajectories = cms.InputTag("thfilter"),
    pixelClusters = cms.InputTag("thClusters"),
    Common = cms.PSet(
        maxChi2 = cms.double(30.0)
    ),
    stripClusters = cms.InputTag("thClusters")
)


#TRACKER HITS
import RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi
import RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi
fourthPixelRecHits = RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi.siPixelRecHits.clone()
fourthStripRecHits = RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi.siStripMatchedRecHits.clone()
fourthPixelRecHits.src = 'fourthClusters'
fourthStripRecHits.ClusterProducer = 'fourthClusters'




#SEEDS
import RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cff
fourthPLSeeds = RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cff.globalMixedSeeds.clone()
import RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi
fourthPLSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'FourthLayerPairs'
fourthPLSeeds.RegionFactoryPSet.RegionPSet.ptMin = 0.6
fourthPLSeeds.RegionFactoryPSet.RegionPSet.originHalfLength = 10.0
fourthPLSeeds.RegionFactoryPSet.RegionPSet.originRadius = 2.0


#TRAJECTORY MEASUREMENT
fourthMeasurementTracker = RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi.MeasurementTracker.clone()
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi
fourthMeasurementTracker.ComponentName = 'fourthMeasurementTracker'
fourthMeasurementTracker.pixelClusterProducer = 'fourthClusters'
fourthMeasurementTracker.stripClusterProducer = 'fourthClusters'

#TRAJECTORY FILTER
fourthCkfTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone()
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
fourthCkfTrajectoryFilter.ComponentName = 'fourthCkfTrajectoryFilter'
fourthCkfTrajectoryFilter.filterPset.maxLostHits = 0
fourthCkfTrajectoryFilter.filterPset.minimumNumberOfHits = 5
fourthCkfTrajectoryFilter.filterPset.minPt = 0.3

#TRAJECTORY BUILDER
fourthCkfTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone()
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
fourthCkfTrajectoryBuilder.ComponentName = 'fourthCkfTrajectoryBuilder'
fourthCkfTrajectoryBuilder.MeasurementTrackerName = 'fourthMeasurementTracker'
fourthCkfTrajectoryBuilder.trajectoryFilterName = 'fourthCkfTrajectoryFilter'
fourthCkfTrajectoryBuilder.minNrOfHitsForRebuild = 5


#TRACK CANDIDATES
fourthTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone()
import RecoTracker.TrackProducer.TrackProducer_cfi
fourthTrackCandidates.src = cms.InputTag('fourthPLSeeds')
fourthTrackCandidates.TrajectoryBuilder = 'fourthCkfTrajectoryBuilder'


#TRACKS
fourthWithMaterialTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone()
fourthWithMaterialTracks.src = 'fourthTrackCandidates'
fourthWithMaterialTracks.clusterRemovalInfo = 'fourthClusters'
fourthWithMaterialTracks.AlgorithmName = cms.string('iter4') 

#SEEDING LAYERS
fourthlayerpairs = cms.ESProducer("PixelLessLayerPairsESProducer",
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
        useRingSlector = cms.untracked.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        minRing = cms.int32(1),
        maxRing = cms.int32(2)
    ),
    TEC = cms.PSet(
        matchedRecHits = cms.InputTag("fourthStripRecHits","matchedRecHit"),
        useRingSlector = cms.untracked.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        minRing = cms.int32(1),
        maxRing = cms.int32(2)
    )
)


# track selection
import RecoTracker.FinalTrackSelectors.selectLoose_cfi
import RecoTracker.FinalTrackSelectors.selectTight_cfi
import RecoTracker.FinalTrackSelectors.selectHighPurity_cfi
import RecoTracker.FinalTrackSelectors.ctfrsTrackListMerger_cfi

pixellessStepLoose = RecoTracker.FinalTrackSelectors.selectLoose_cfi.selectLoose.clone()
pixellessStepLoose.src = 'fourthWithMaterialTracks'
pixellessStepLoose.keepAllTracks = False
pixellessStepLoose.copyExtras = True
pixellessStepLoose.copyTrajectories = True
pixellessStepLoose.chi2n_par = 0.6
pixellessStepLoose.res_par = ( 0.003, 0.001 )
pixellessStepLoose.minNumberLayers = 5
pixellessStepLoose.d0_par1 = ( 1.5, 4.0 )
pixellessStepLoose.dz_par1 = ( 1.5, 4.0 )
pixellessStepLoose.d0_par2 = ( 1.5, 4.0 )
pixellessStepLoose.dz_par2 = ( 1.5, 4.0 )

pixellessStepTight = RecoTracker.FinalTrackSelectors.selectTight_cfi.selectTight.clone()
pixellessStepTight.src = 'pixellessStepLoose'
pixellessStepTight.keepAllTracks = True
pixellessStepTight.copyExtras = True
pixellessStepTight.copyTrajectories = True
pixellessStepTight.chi2n_par = 0.4
pixellessStepTight.res_par = ( 0.003, 0.001 )
pixellessStepTight.minNumberLayers = 5
pixellessStepTight.d0_par1 = ( 1.1, 4.0 )
pixellessStepTight.dz_par1 = ( 1.1, 4.0 )
pixellessStepTight.d0_par2 = ( 1.1, 4.0 )
pixellessStepTight.dz_par2 = ( 1.1, 4.0 )

pixellessStep = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone()
pixellessStep.src = 'pixellessStepTight'
pixellessStep.keepAllTracks = True
pixellessStep.copyExtras = True
pixellessStep.copyTrajectories = True
pixellessStep.chi2n_par = 0.3
pixellessStep.res_par = ( 0.003, 0.001 )
pixellessStep.minNumberLayers = 5
pixellessStep.d0_par1 = ( 1.0, 4.0 )
pixellessStep.dz_par1 = ( 1.0, 4.0 )
pixellessStep.d0_par2 = ( 1.0, 4.0 )
pixellessStep.dz_par2 = ( 1.0, 4.0 )

fourthStep = cms.Sequence(thfilter*fourthClusters*
                          fourthPixelRecHits*fourthStripRecHits*
                          fourthPLSeeds*
                          fourthTrackCandidates*
                          fourthWithMaterialTracks*
                          pixellessStepLoose*
                          pixellessStepTight*
                          pixellessStep)
