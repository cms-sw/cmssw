import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.AlpakaCore.functions import makeSerialClone

hltHgcalSoARecHitsLayerClustersProducerHSci = cms.EDProducer("HGCalCLUEsteringLayerClustersProducer@alpaka",
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    ),
    hgcalRecHitsSoA = cms.InputTag("hltHgcalSoARecHitsProducerHSci"),
    detector = cms.string('BH'),
    deltac = cms.float(0.0315),
    kappa = cms.float(9),
    outlierDeltaFactor = cms.float(2.0)
)

hltHgcalSoARecHitsLayerClustersProducerHSciSerialSync = makeSerialClone(hltHgcalSoARecHitsLayerClustersProducerHSci,
                                                                    hgcalRecHitsSoA = "hltHgcalSoARecHitsProducerHSciSerialSync"
)
