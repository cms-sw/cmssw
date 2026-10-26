import FWCore.ParameterSet.Config as cms

hltHgCalLayerClustersFromSoAProducerL1Seeded = cms.EDProducer("HGCalLayerClustersFromSoAProducer",
    detector = cms.string('EE'),
    hgcalRecHitsLayerClustersSoA = cms.InputTag("hltHgcalSoARecHitsLayerClustersProducerL1Seeded"),
    hgcalRecHitsSoA = cms.InputTag("hltHgcalSoARecHitsProducerL1Seeded"),
    src = cms.InputTag("hltHgcalSoALayerClustersProducerL1Seeded"),
    timeClname = cms.string('timeLayerCluster')
)
