import FWCore.ParameterSet.Config as cms

hltHgCalLayerClustersFromSoAProducerHSciL1Seeded = cms.EDProducer("HGCalLayerClustersFromSoAProducer",
    detector = cms.string('BH'),
    hgcalRecHitsLayerClustersSoA = cms.InputTag("hltHgcalSoARecHitsLayerClustersProducerHSciL1Seeded"),
    hgcalRecHitsSoA = cms.InputTag("hltHgcalSoARecHitsProducerHSciL1Seeded"),
    src = cms.InputTag("hltHgcalSoALayerClustersProducerHSciL1Seeded"),
    timeClname = cms.string('timeLayerCluster')
)
