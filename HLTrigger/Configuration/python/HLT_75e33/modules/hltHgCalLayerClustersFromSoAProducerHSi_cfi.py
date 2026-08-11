import FWCore.ParameterSet.Config as cms

hltHgCalLayerClustersFromSoAProducerHSi = cms.EDProducer("HGCalLayerClustersFromSoAProducer",
    detector = cms.string('FH'),
    hgcalRecHitsLayerClustersSoA = cms.InputTag("hltHgcalSoARecHitsLayerClustersProducerHSi"),
    hgcalRecHitsSoA = cms.InputTag("hltHgcalSoARecHitsProducerHSi"),
    nHitsTime = cms.uint32(3),
    src = cms.InputTag("hltHgcalSoALayerClustersProducerHSi"),
    timeClname = cms.string('timeLayerCluster')
)

hltHgCalLayerClustersFromSoAProducerHSiSerialSync = cms.EDProducer("HGCalLayerClustersFromSoAProducer",
    detector = cms.string('FH'),
    hgcalRecHitsLayerClustersSoA = cms.InputTag("hltHgcalSoARecHitsLayerClustersProducerHSiSerialSync"),
    hgcalRecHitsSoA = cms.InputTag("hltHgcalSoARecHitsProducerHSiSerialSync"),
    nHitsTime = cms.uint32(3),
    src = cms.InputTag("hltHgcalSoALayerClustersProducerHSiSerialSync"),
    timeClname = cms.string('timeLayerCluster')
)
