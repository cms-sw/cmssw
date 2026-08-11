import FWCore.ParameterSet.Config as cms

hltHgcalSoARecHitsLayerClustersProducerL1Seeded = cms.EDProducer("HGCalCLUEsteringLayerClustersProducer@alpaka",
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    ),
    hgcalRecHitsSoA = cms.InputTag("hltHgcalSoARecHitsProducerL1Seeded"),
    detector = cms.string('EE'),
    deltac = cms.float(1.3),
    kappa = cms.float(9),
    outlierDeltaFactor = cms.float(2.0)
)
