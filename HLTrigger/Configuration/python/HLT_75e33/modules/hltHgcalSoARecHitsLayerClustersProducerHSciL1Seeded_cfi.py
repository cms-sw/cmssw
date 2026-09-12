import FWCore.ParameterSet.Config as cms

hltHgcalSoARecHitsLayerClustersProducerHSciL1Seeded = cms.EDProducer("HGCalCLUEsteringLayerClustersProducer@alpaka",
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    ),
    hgcalRecHitsSoA = cms.InputTag("hltHgcalSoARecHitsProducerHSciL1Seeded"),
    detector = cms.string('BH'),
    deltac = cms.double(0.0315),
    kappa = cms.double(9),
    outlierDeltaFactor = cms.double(2.0)
)
