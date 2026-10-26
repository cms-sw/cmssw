import FWCore.ParameterSet.Config as cms

hltHgCalLayerClustersFromSoAProducer = cms.EDProducer("HGCalLayerClustersFromSoAProducer",
    detector = cms.string('EE'),
    hgcalRecHitsLayerClustersSoA = cms.InputTag("hltHgcalSoARecHitsLayerClustersProducer"),
    hgcalRecHitsSoA = cms.InputTag("hltHgcalSoARecHitsProducer"),
    src = cms.InputTag("hltHgcalSoALayerClustersProducer"),
    timeClname = cms.string('timeLayerCluster')
)

hltHgCalLayerClustersFromSoAProducerSerialSync = cms.EDProducer("HGCalLayerClustersFromSoAProducer",
    detector = cms.string('EE'),
    hgcalRecHitsLayerClustersSoA = cms.InputTag("hltHgcalSoARecHitsLayerClustersProducerSerialSync"),
    hgcalRecHitsSoA = cms.InputTag("hltHgcalSoARecHitsProducerSerialSync"),
    src = cms.InputTag("hltHgcalSoALayerClustersProducerSerialSync"),
    timeClname = cms.string('timeLayerCluster')
)
