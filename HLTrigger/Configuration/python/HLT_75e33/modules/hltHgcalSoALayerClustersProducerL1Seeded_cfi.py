import FWCore.ParameterSet.Config as cms

hltHgcalSoALayerClustersProducerL1Seeded = cms.EDProducer("HGCalSoALayerClustersProducer@alpaka",
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    ),
    hgcalRecHitsLayerClustersSoA = cms.InputTag("hltHgcalSoARecHitsLayerClustersProducerL1Seeded"),
    hgcalRecHitsSoA = cms.InputTag("hltHgcalSoARecHitsProducerL1Seeded"),
    detector = cms.string('EE'),
    positionDeltaRho2 = cms.double(1.69),
    thresholdW0 = cms.double(2.9)
)
