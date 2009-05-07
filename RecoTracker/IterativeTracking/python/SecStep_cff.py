import FWCore.ParameterSet.Config as cms

# new hit collection
secClusters = cms.EDFilter("TrackClusterRemover",
    trajectories = cms.InputTag("firstfilter"),
    pixelClusters = cms.InputTag("siPixelClusters"),
    Common = cms.PSet(
        maxChi2 = cms.double(30.0)
    ),
    stripClusters = cms.InputTag("siStripClusters")
)

import RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi
secPixelRecHits = RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi.siPixelRecHits.clone()
import RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi
secStripRecHits = RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi.siStripMatchedRecHits.clone()

secPixelRecHits.src = cms.InputTag("secClusters")
secStripRecHits.ClusterProducer = 'secClusters'


# seeding

import RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi
seclayertriplets = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.pixellayertriplets.clone()
import RecoTracker.TkSeedGenerator.GlobalSeedsFromTripletsWithVertices_cfi
secTriplets = RecoTracker.TkSeedGenerator.GlobalSeedsFromTripletsWithVertices_cfi.globalSeedsFromTripletsWithVertices.clone()

seclayertriplets.ComponentName = 'SecLayerTriplets'
seclayertriplets.BPix.HitProducer = 'secPixelRecHits'
seclayertriplets.FPix.HitProducer = 'secPixelRecHits'
secTriplets.RegionFactoryPSet.RegionPSet.originHalfLength = 17.5
secTriplets.OrderedHitsFactoryPSet.SeedingLayers = 'SecLayerTriplets'
secTriplets.RegionFactoryPSet.RegionPSet.ptMin = 0.3


# building 
import RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi
secMeasurementTracker = RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi.MeasurementTracker.clone()
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi
secCkfTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone()
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
secCkfTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone()
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
secTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone()

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


# fitting
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
secWithMaterialTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
secWithMaterialTracks.src = 'secTrackCandidates'
secWithMaterialTracks.clusterRemovalInfo = 'secClusters'


# track selection
from RecoTracker.IterativeTracking.SecVxFilter_cff import *

secondStep = cms.Sequence(secClusters*secPixelRecHits*secStripRecHits*secTriplets*secTrackCandidates*secWithMaterialTracks*secStep)
