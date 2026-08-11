import FWCore.ParameterSet.Config as cms

hltHgcalSoARecHitsLayerClustersProducerL1Seeded = cms.EDProducer("HGCalCLUEsteringLayerClustersProducer@alpaka",
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    ),
    hgcalRecHitsSoA = cms.InputTag("hltHgcalSoARecHitsProducerL1Seeded"),
    detector = cms.string('EE'),
    deltac = cms.double(1.3),
    kappa = cms.double(9),
    outlierDeltaFactor = cms.double(2.0)
)
