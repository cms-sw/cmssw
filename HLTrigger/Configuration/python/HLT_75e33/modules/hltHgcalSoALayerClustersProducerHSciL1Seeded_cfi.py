import FWCore.ParameterSet.Config as cms

hltHgcalSoALayerClustersProducerHSciL1Seeded = cms.EDProducer("HGCalSoALayerClustersProducer@alpaka",
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    ),
    hgcalRecHitsLayerClustersSoA = cms.InputTag("hltHgcalSoARecHitsLayerClustersProducerHSciL1Seeded"),
    hgcalRecHitsSoA = cms.InputTag("hltHgcalSoARecHitsProducerHSciL1Seeded"),
    detector = cms.string('BH'),
    positionDeltaRho2 = cms.double(1.69),
    thresholdW0 = cms.double(2.9)
)
