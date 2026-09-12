import FWCore.ParameterSet.Config as cms

hltHgcalSoALayerClustersProducerHSiL1Seeded = cms.EDProducer("HGCalSoALayerClustersProducer@alpaka",
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    ),
    hgcalRecHitsLayerClustersSoA = cms.InputTag("hltHgcalSoARecHitsLayerClustersProducerHSiL1Seeded"),
    hgcalRecHitsSoA = cms.InputTag("hltHgcalSoARecHitsProducerHSiL1Seeded"),
    detector = cms.string('FH'),
    positionDeltaRho2 = cms.double(1.69),
    thresholdW0 = cms.double(2.9)
)
