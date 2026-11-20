import FWCore.ParameterSet.Config as cms

hltHgcalSoARecHitsLayerClustersProducerHSiL1Seeded = cms.EDProducer("HGCalCLUEsteringLayerClustersProducer@alpaka",
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    ),
    hgcalRecHitsSoA = cms.InputTag("hltHgcalSoARecHitsProducerHSiL1Seeded"),
    detector = cms.string('FH'),
    deltac = cms.float(1.3),
    kappa = cms.float(9),
    outlierDeltaFactor = cms.float(2.0)
)
