import FWCore.ParameterSet.Config as cms

hltHgCalLayerClustersFromSoAProducerHSci = cms.EDProducer("HGCalLayerClustersFromSoAProducer",
    detector = cms.string('BH'),
    hgcalRecHitsLayerClustersSoA = cms.InputTag("hltHgcalSoARecHitsLayerClustersProducerHSci"),
    hgcalRecHitsSoA = cms.InputTag("hltHgcalSoARecHitsProducerHSci"),
    nHitsTime = cms.uint32(3),
    src = cms.InputTag("hltHgcalSoALayerClustersProducerHSci"),
    timeClname = cms.string('timeLayerCluster')
)

hltHgCalLayerClustersFromSoAProducerHSciSerialSync = cms.EDProducer("HGCalLayerClustersFromSoAProducer",
    detector = cms.string('BH'),
    hgcalRecHitsLayerClustersSoA = cms.InputTag("hltHgcalSoARecHitsLayerClustersProducerHSciSerialSync"),
    hgcalRecHitsSoA = cms.InputTag("hltHgcalSoARecHitsProducerHSciSerialSync"),
    nHitsTime = cms.uint32(3),
    src = cms.InputTag("hltHgcalSoALayerClustersProducerHSciSerialSync"),
    timeClname = cms.string('timeLayerCluster')
)
