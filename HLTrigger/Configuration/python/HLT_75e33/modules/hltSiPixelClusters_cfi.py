import FWCore.ParameterSet.Config as cms

hltSiPixelClusters = cms.EDProducer('SiPixelDigisClustersFromSoAAlpakaPhase2',
    src = cms.InputTag('hltPhase2SiPixelClustersSoA'),
    clusterThreshold_layer1 = cms.int32(4000),
    clusterThreshold_otherLayers = cms.int32(4000),
    produceDigis = cms.bool(False),
    storeDigis = cms.bool(False)
)
