import FWCore.ParameterSet.Config as cms

hltHgCalLayerClustersFromSoAProducerHSi = cms.EDProducer("HGCalLayerClustersFromSoAProducer",
    detector = cms.string('FH'),
    hgcalRecHitsLayerClustersSoA = cms.InputTag("hltHgcalSoARecHitsLayerClustersProducerHSi"),
    hgcalRecHitsSoA = cms.InputTag("hltHgcalSoARecHitsProducerHSi"),
    src = cms.InputTag("hltHgcalSoALayerClustersProducerHSi"),
    timeClname = cms.string('timeLayerCluster')
)

hltHgCalLayerClustersFromSoAProducerHSiSerialSync = cms.EDProducer("HGCalLayerClustersFromSoAProducer",
    detector = cms.string('FH'),
    hgcalRecHitsLayerClustersSoA = cms.InputTag("hltHgcalSoARecHitsLayerClustersProducerHSiSerialSync"),
    hgcalRecHitsSoA = cms.InputTag("hltHgcalSoARecHitsProducerHSiSerialSync"),
    src = cms.InputTag("hltHgcalSoALayerClustersProducerHSiSerialSync"),
    timeClname = cms.string('timeLayerCluster')
)
