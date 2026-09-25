import FWCore.ParameterSet.Config as cms

hltHgCalLayerClustersFromSoAProducerHSiL1Seeded = cms.EDProducer("HGCalLayerClustersFromSoAProducer",
    detector = cms.string('FH'),
    hgcalRecHitsLayerClustersSoA = cms.InputTag("hltHgcalSoARecHitsLayerClustersProducerHSiL1Seeded"),
    hgcalRecHitsSoA = cms.InputTag("hltHgcalSoARecHitsProducerHSiL1Seeded"),
    nHitsTime = cms.uint32(3),
    src = cms.InputTag("hltHgcalSoALayerClustersProducerHSiL1Seeded"),
    timeClname = cms.string('timeLayerCluster')
)
