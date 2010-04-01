import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFTracking.nuclear_cff import *
import RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi
nuclearPixelRecHits = RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi.siPixelRecHits.clone()
import RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi
nuclearStripRecHits = RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi.siStripMatchedRecHits.clone()
import RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi
nuclearMeasurementTracker = RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi.MeasurementTracker.clone()
nuclearClusters = cms.EDProducer("TrackClusterRemover",
    oldClusterRemovalInfo = cms.InputTag("thClusters"),
    trajectories = cms.InputTag("thStep"),
    pixelClusters = cms.InputTag("thClusters"),
    Common = cms.PSet(
        maxChi2 = cms.double(30.0)
    ),
    stripClusters = cms.InputTag("thClusters")
)

nuclearRemainingHits = cms.Sequence(nuclearClusters*nuclearPixelRecHits*nuclearStripRecHits*nuclear)
nuclearPixelRecHits.src = 'nuclearClusters'
nuclearStripRecHits.ClusterProducer = 'nuclearClusters'
nuclearMeasurementTracker.ComponentName = 'nuclearMeasurementTracker'
nuclearMeasurementTracker.pixelClusterProducer = 'nuclearClusters'
nuclearMeasurementTracker.stripClusterProducer = 'nuclearClusters'
firstnuclearSeed.MeasurementTrackerName = 'nuclearMeasurementTracker'
nuclearCkfTrajectoryBuilder.MeasurementTrackerName = 'nuclearMeasurementTracker'
