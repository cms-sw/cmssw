import FWCore.ParameterSet.Config as cms

siPixelClustersTop = cms.EDProducer("PixelClusterSelectorTopBottom",
    label = cms.InputTag("siPixelClusters"),
    y = cms.double(+1)
)
siPixelClustersBottom = cms.EDProducer("PixelClusterSelectorTopBottom",
    label = cms.InputTag("siPixelClusters"),
    y = cms.double(-1)
)
siStripClustersTop = cms.EDProducer("StripClusterSelectorTopBottom",
    label = cms.InputTag("siStripClusters"),
    y = cms.double(+1)
)
siStripClustersBottom = cms.EDProducer("StripClusterSelectorTopBottom",
    label = cms.InputTag("siStripClusters"),
    y = cms.double(-1)
)
