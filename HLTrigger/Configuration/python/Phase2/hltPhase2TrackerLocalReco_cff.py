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

hltPhase2siPhase2Clusters = _siPhase2Clusters.clone()

hltPhase2siPixelClusters = _SiPixelClusterizerDefault.clone(
    ElectronPerADCGain=600.0,
    MissCalibrate=False,
    Phase2Calibration=True,
    VCaltoElectronGain=1,
    VCaltoElectronGain_L1=1,
    VCaltoElectronOffset=0,
    VCaltoElectronOffset_L1=0,
    src=cms.InputTag("simSiPixelDigis", "Pixel"),
)

hltPhase2siPixelClusterShapeCache = _siPixelClusterShapeCache.clone(
    src=cms.InputTag("hltPhase2siPixelClusters")
)

hltPhase2siPixelRecHits = _siPixelRecHits.clone(
    src=cms.InputTag("hltPhase2siPixelClusters")
)

hltPhase2MeasurementTrackerEvent = _measurementTrackerEventDefault.clone(
    Phase2TrackerCluster1DProducer="hltPhase2siPhase2Clusters",
    badPixelFEDChannelCollectionLabels=cms.VInputTag("siPixelDigis"),
    inactivePixelDetectorLabels=cms.VInputTag(),
    pixelClusterProducer=cms.string("hltPhase2siPixelClusters"),
    stripClusterProducer="",
)

hltPhase2trackerClusterCheck = _trackerClusterCheckDefault.clone(doClusterCheck=False)

hltPhase2TrackerLocalRecoSequence = cms.Sequence(
    hltPhase2siPixelClusters
    + hltPhase2siPixelClusterShapeCache
    + hltPhase2siPixelRecHits
    + hltPhase2siPhase2Clusters
    + hltPhase2MeasurementTrackerEvent
    + hltPhase2trackerClusterCheck
)
