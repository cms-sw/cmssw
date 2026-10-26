import FWCore.ParameterSet.Config as cms

hltHgCalLayerClustersFromSoAProducerHSci = cms.EDProducer("HGCalLayerClustersFromSoAProducer",
    detector = cms.string('BH'),
    hgcalRecHitsLayerClustersSoA = cms.InputTag("hltHgcalSoARecHitsLayerClustersProducerHSci"),
    hgcalRecHitsSoA = cms.InputTag("hltHgcalSoARecHitsProducerHSci"),
    src = cms.InputTag("hltHgcalSoALayerClustersProducerHSci"),
    timeClname = cms.string('timeLayerCluster')
)

hltHgCalLayerClustersFromSoAProducerHSciSerialSync = cms.EDProducer("HGCalLayerClustersFromSoAProducer",
    detector = cms.string('BH'),
    hgcalRecHitsLayerClustersSoA = cms.InputTag("hltHgcalSoARecHitsLayerClustersProducerHSciSerialSync"),
    hgcalRecHitsSoA = cms.InputTag("hltHgcalSoARecHitsProducerHSciSerialSync"),
    src = cms.InputTag("hltHgcalSoALayerClustersProducerHSciSerialSync"),
    timeClname = cms.string('timeLayerCluster')
)
