import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiPhase2Clusterizer.phase2TrackerClusterizer_cfi import (
    siPhase2Clusters as _siPhase2Clusters,
)
from RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizerDefault_cfi import (
    SiPixelClusterizerDefault as _SiPixelClusterizerDefault,
)
from RecoPixelVertexing.PixelLowPtUtilities.siPixelClusterShapeCache_cfi import (
    siPixelClusterShapeCache as _siPixelClusterShapeCache,
)
from RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi import (
    siPixelRecHits as _siPixelRecHits,
)
from RecoTracker.MeasurementDet.measurementTrackerEventDefault_cfi import (
    measurementTrackerEventDefault as _measurementTrackerEventDefault,
)
from RecoTracker.TkSeedGenerator.trackerClusterCheckDefault_cfi import (
    trackerClusterCheckDefault as _trackerClusterCheckDefault,
)

siPhase2Clusters = _siPhase2Clusters.clone()

siPixelClusters = _SiPixelClusterizerDefault.clone(
    ElectronPerADCGain=600.0,
    MissCalibrate=False,
    Phase2Calibration=True,
    VCaltoElectronGain=1,
    VCaltoElectronGain_L1=1,
    VCaltoElectronOffset=0,
    VCaltoElectronOffset_L1=0,
    src=cms.InputTag("simSiPixelDigis", "Pixel"),
)

siPixelClusterShapeCache = _siPixelClusterShapeCache.clone()

siPixelRecHits = _siPixelRecHits.clone()

MeasurementTrackerEvent = _measurementTrackerEventDefault.clone(
    Phase2TrackerCluster1DProducer="siPhase2Clusters",
    badPixelFEDChannelCollectionLabels=cms.VInputTag("siPixelDigis"),
    inactivePixelDetectorLabels=cms.VInputTag(),
    stripClusterProducer="",
)

trackerClusterCheck = _trackerClusterCheckDefault.clone(doClusterCheck=False)

hltPhase2TrackerLocalRecoSequence = cms.Sequence(
    siPixelClusters
    + siPixelClusterShapeCache
    + siPixelRecHits
    + siPhase2Clusters
    + MeasurementTrackerEvent
    + trackerClusterCheck
)
