import FWCore.ParameterSet.Config as cms

#
# Very large impact parameter tracking using TOB + TEC ring 5 seeding
#


#HIT REMOVAL
fifthClusters = cms.EDFilter("TrackClusterRemover",
    oldClusterRemovalInfo = cms.InputTag("fourthClusters"),
    trajectories = cms.InputTag("fourthStep"),
    pixelClusters = cms.InputTag("fourthClusters"),
    stripClusters = cms.InputTag("fourthClusters"),
    Common = cms.PSet(
        maxChi2 = cms.double(30.0)
# To debug this tracking iteration, you can run it on all hits,
# as opposed to only those not used by previous iterations, with this ...
#       maxChi2 = cms.double(0.0)
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
import RecoTracker.TkSeedingLayers.TobTecLayerPairs_cfi
fifthtobteclayerpairs = RecoTracker.TkSeedingLayers.TobTecLayerPairs_cfi.tobteclayerpairs.clone()
fifthtobteclayerpairs.ComponentName = 'fifthTobTecLayerPairs'
fifthtobteclayerpairs.TOB.matchedRecHits = 'fifthStripRecHits:matchedRecHit'
fifthtobteclayerpairs.TEC.matchedRecHits = 'fifthStripRecHits:matchedRecHit'

#SEEDS
import RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cfi
fifthSeeds = RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cfi.globalMixedSeeds.clone()
fifthSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'fifthTobTecLayerPairs'
fifthSeeds.RegionFactoryPSet.RegionPSet.ptMin = 0.9
fifthSeeds.RegionFactoryPSet.RegionPSet.originHalfLength = 30.0
fifthSeeds.RegionFactoryPSet.RegionPSet.originRadius = 20.0

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
#fifthCkfTrajectoryFilter.filterPset.maxLostHits = 1
#fifthCkfTrajectoryFilter.filterPset.maxConsecLostHits = 2
fifthCkfTrajectoryFilter.filterPset.minimumNumberOfHits = 6
fifthCkfTrajectoryFilter.filterPset.minPt = 0.9
fifthCkfTrajectoryFilter.filterPset.minHitsMinPt = 3

fifthCkfInOutTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone()
fifthCkfInOutTrajectoryFilter.ComponentName = 'fifthCkfInOutTrajectoryFilter'
#fifthCkfInOutTrajectoryFilter.filterPset.maxLostHits = 1
#fifthCkfInOutTrajectoryFilter.filterPset.maxConsecLostHits = 2
fifthCkfInOutTrajectoryFilter.filterPset.minimumNumberOfHits = 4
fifthCkfInOutTrajectoryFilter.filterPset.minPt = 0.9
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
#fifthCkfTrajectoryBuilder.maxCand = 5
#fifthCkfTrajectoryBuilder.lostHitPenalty = 100.
#fifthCkfTrajectoryBuilder.alwaysUseInvalidHits = False

#TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
fifthTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone()
fifthTrackCandidates.src = cms.InputTag('fifthSeeds')
fifthTrackCandidates.TrajectoryBuilder = 'fifthCkfTrajectoryBuilder'
fifthTrackCandidates.doSeedingRegionRebuilding = True
fifthTrackCandidates.useHitsSplitting = True
fifthTrackCandidates.cleanTrajectoryAfterInOut = False

#
# TRACK FITTING AND SMOOTHING
#

import TrackingTools.TrackFitters.RungeKuttaKFFittingSmootherESProducer_cfi
fifthFittingSmootherWithOutlierRejection = TrackingTools.TrackFitters.RungeKuttaKFFittingSmootherESProducer_cfi.RKFittingSmoother.clone()
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
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
fifthWithMaterialTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
fifthWithMaterialTracks.src = 'fifthTrackCandidates'
fifthWithMaterialTracks.clusterRemovalInfo = 'fifthClusters'
fifthWithMaterialTracks.AlgorithmName = cms.string('iterFifth')
fifthWithMaterialTracks.Fitter = 'fifthFittingSmootherWithOutlierRejection'

fifthStep = cms.Sequence(fifthClusters*
                          fifthPixelRecHits*fifthStripRecHits*
                          fifthSeeds*
                          fifthTrackCandidates*
                          fifthWithMaterialTracks)
                          







