import FWCore.ParameterSet.Config as cms

hltSiPixelClusters = cms.EDProducer("SiPixelClusterProducer",
    ChannelThreshold = cms.int32(1000),
    ClusterMode = cms.string('PixelThresholdClusterizer'),
    ClusterThreshold = cms.int32(4000),
    ClusterThreshold_L1 = cms.int32(4000),
    ElectronPerADCGain = cms.double(1500.0),
    MissCalibrate = cms.bool(False),
    Phase2Calibration = cms.bool(True),
    Phase2DigiBaseline = cms.double(1000.0),
    Phase2KinkADC = cms.int32(8),
    Phase2ReadoutMode = cms.int32(3),
    SeedThreshold = cms.int32(1000),
    SplitClusters = cms.bool(False),
    VCaltoElectronGain = cms.int32(1),
    VCaltoElectronGain_L1 = cms.int32(1),
    VCaltoElectronOffset = cms.int32(0),
    VCaltoElectronOffset_L1 = cms.int32(0),
    maxNumberOfClusters = cms.int32(-1),
    mightGet = cms.optional.untracked.vstring,
    payloadType = cms.string('None'),
    src = cms.InputTag("simSiPixelDigis","Pixel")
)

from Configuration.ProcessModifiers.alpaka_cff import alpaka
_hltSiPixelClusters = cms.EDProducer('SiPixelDigisClustersFromSoAAlpakaPhase2',
    src = cms.InputTag('hltPhase2SiPixelClustersSoA'),
    clusterThreshold_layer1 = cms.int32(4000),
    clusterThreshold_otherLayers = cms.int32(4000),
    produceDigis = cms.bool(False),
    storeDigis = cms.bool(False)
)
alpaka.toReplaceWith(hltSiPixelClusters, _hltSiPixelClusters)
