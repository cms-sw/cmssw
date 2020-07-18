import FWCore.ParameterSet.Config as cms

siPhase2Clusters = cms.EDProducer(
    "Phase2TrackerClusterizer",
    maxClusterSize=cms.uint32(0),
    maxNumberClusters=cms.uint32(0),
    src=cms.InputTag("mix", "Tracker"),
)

siPixelClusters = cms.EDProducer(
    "SiPixelClusterProducer",
    ChannelThreshold=cms.int32(1000),
    ClusterMode=cms.string("PixelThresholdClusterizer"),
    ClusterThreshold=cms.int32(4000),
    ClusterThreshold_L1=cms.int32(4000),
    ElectronPerADCGain=cms.double(600.0),
    MissCalibrate=cms.bool(False),
    Phase2Calibration=cms.bool(True),
    Phase2DigiBaseline=cms.double(1200),
    Phase2KinkADC=cms.int32(8),
    Phase2ReadoutMode=cms.int32(-1),
    SeedThreshold=cms.int32(1000),
    SplitClusters=cms.bool(False),
    VCaltoElectronGain=cms.int32(1),
    VCaltoElectronGain_L1=cms.int32(1),
    VCaltoElectronOffset=cms.int32(0),
    VCaltoElectronOffset_L1=cms.int32(0),
    maxNumberOfClusters=cms.int32(-1),
    mightGet=cms.optional.untracked.vstring,
    payloadType=cms.string("Offline"),
    src=cms.InputTag("simSiPixelDigis", "Pixel"),
)

siPixelClusterShapeCache = cms.EDProducer(
    "SiPixelClusterShapeCacheProducer",
    mightGet=cms.optional.untracked.vstring,
    onDemand=cms.bool(False),
    src=cms.InputTag("siPixelClusters"),
)

siPixelRecHits = cms.EDProducer(
    "SiPixelRecHitConverter",
    CPE=cms.string("PixelCPEGeneric"),
    VerboseLevel=cms.untracked.int32(0),
    src=cms.InputTag("siPixelClusters"),
)

MeasurementTrackerEvent = cms.EDProducer(
    "MeasurementTrackerEventProducer",
    Phase2TrackerCluster1DProducer=cms.string("siPhase2Clusters"),
    badPixelFEDChannelCollectionLabels=cms.VInputTag("siPixelDigis"),
    inactivePixelDetectorLabels=cms.VInputTag(),
    inactiveStripDetectorLabels=cms.VInputTag("siStripDigis"),
    measurementTracker=cms.string(""),
    mightGet=cms.optional.untracked.vstring,
    pixelCablingMapLabel=cms.string(""),
    pixelClusterProducer=cms.string("siPixelClusters"),
    skipClusters=cms.InputTag(""),
    stripClusterProducer=cms.string(""),
    switchOffPixelsIfEmpty=cms.bool(True),
)

trackerClusterCheck = cms.EDProducer(
    "ClusterCheckerEDProducer",
    ClusterCollectionLabel=cms.InputTag("siStripClusters"),
    MaxNumberOfCosmicClusters=cms.uint32(400000),
    MaxNumberOfPixelClusters=cms.uint32(40000),
    PixelClusterCollectionLabel=cms.InputTag("siPixelClusters"),
    cut=cms.string(
        "strip < 400000 && pixel < 40000 && (strip < 50000 + 10*pixel) && (pixel < 5000 + 0.1*strip)"
    ),
    doClusterCheck=cms.bool(False),
    mightGet=cms.optional.untracked.vstring,
    silentClusterCheck=cms.untracked.bool(False),
)

trackerLocalRecoSequence = cms.Sequence(
    siPixelClusters
    + siPixelClusterShapeCache
    + siPixelRecHits
    + siPhase2Clusters
    + MeasurementTrackerEvent
    + trackerClusterCheck
)
