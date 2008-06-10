import FWCore.ParameterSet.Config as cms

import RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi
secPixelRecHits = RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi.siPixelRecHits.clone()
import RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi
secStripRecHits = RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi.siStripMatchedRecHits.clone()
import RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi
seclayertriplets = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.pixellayertriplets.clone()
import RecoTracker.TkSeedGenerator.GlobalSeedsFromTripletsWithVertices_cfi
secTriplets = RecoTracker.TkSeedGenerator.GlobalSeedsFromTripletsWithVertices_cfi.globalSeedsFromTripletsWithVertices.clone()
import RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi
secMeasurementTracker = RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi.MeasurementTracker.clone()
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi
secCkfTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone()
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
secCkfTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone()
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
secTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone()
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
secWithMaterialTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
from RecoTracker.IterativeTracking.SecVxFilter_cff import *
secClusters = cms.EDFilter("TrackClusterRemover",
    trajectories = cms.InputTag("firstfilter"),
    pixelClusters = cms.InputTag("siPixelClusters"),
    Common = cms.PSet(
        maxChi2 = cms.double(30.0)
    ),
    stripClusters = cms.InputTag("siStripClusters")
)

secondStep = cms.Sequence(secClusters*secPixelRecHits*secStripRecHits*secTriplets*secTrackCandidates*secWithMaterialTracks*secStep)
secPixelRecHits.src = cms.InputTag("secClusters")
secStripRecHits.ClusterProducer = 'secClusters'
seclayertriplets.ComponentName = 'SecLayerTriplets'
seclayertriplets.BPix.HitProducer = 'secPixelRecHits'
seclayertriplets.FPix.HitProducer = 'secPixelRecHits'
secTriplets.RegionFactoryPSet.RegionPSet.originHalfLength = 22.7
secTriplets.OrderedHitsFactoryPSet.SeedingLayers = 'SecLayerTriplets'
secTriplets.RegionFactoryPSet.RegionPSet.ptMin = 0.3
secMeasurementTracker.ComponentName = 'secMeasurementTracker'
secMeasurementTracker.pixelClusterProducer = 'secClusters'
secMeasurementTracker.stripClusterProducer = 'secClusters'
secCkfTrajectoryFilter.ComponentName = 'secCkfTrajectoryFilter'
secCkfTrajectoryFilter.filterPset.maxLostHits = 1
secCkfTrajectoryFilter.filterPset.minimumNumberOfHits = 3
secCkfTrajectoryFilter.filterPset.minPt = 0.3
secCkfTrajectoryBuilder.ComponentName = 'secCkfTrajectoryBuilder'
secCkfTrajectoryBuilder.MeasurementTrackerName = 'secMeasurementTracker'
secCkfTrajectoryBuilder.trajectoryFilterName = 'secCkfTrajectoryFilter'
secTrackCandidates.SeedProducer = 'secTriplets'
secTrackCandidates.TrajectoryBuilder = 'secCkfTrajectoryBuilder'
secTrackCandidates.doSeedingRegionRebuilding = True
secTrackCandidates.useHitsSplitting = True
secWithMaterialTracks.src = 'secTrackCandidates'
secWithMaterialTracks.clusterRemovalInfo = 'secClusters'


