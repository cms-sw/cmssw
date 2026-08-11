import FWCore.ParameterSet.Config as cms

hltHgcalSoARecHitsLayerClustersProducerHSiL1Seeded = cms.EDProducer("HGCalCLUEsteringLayerClustersProducer@alpaka",
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    ),
    hgcalRecHitsSoA = cms.InputTag("hltHgcalSoARecHitsProducerHSiL1Seeded"),
    detector = cms.string('FH'),
    deltac = cms.double(1.3),
    kappa = cms.double(9),
    outlierDeltaFactor = cms.double(2.0)
)
