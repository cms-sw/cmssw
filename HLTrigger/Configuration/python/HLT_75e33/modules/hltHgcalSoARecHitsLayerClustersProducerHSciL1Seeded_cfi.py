import FWCore.ParameterSet.Config as cms

hltHgcalSoARecHitsLayerClustersProducerHSciL1Seeded = cms.EDProducer("HGCalCLUEsteringLayerClustersProducer@alpaka",
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    ),
    hgcalRecHitsSoA = cms.InputTag("hltHgcalSoARecHitsProducerHSciL1Seeded"),
    detector = cms.string('BH'),
    deltac = cms.float(0.0315),
    kappa = cms.float(9),
    outlierDeltaFactor = cms.float(2.0)
)
