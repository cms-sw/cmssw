import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.AlpakaCore.functions import makeSerialClone

hltHgcalSoARecHitsLayerClustersProducerHSci = cms.EDProducer("HGCalCLUEsteringLayerClustersProducer@alpaka",
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    ),
    hgcalRecHitsSoA = cms.InputTag("hltHgcalSoARecHitsProducerHSci"),
    detector = cms.string('BH'),
    deltac = cms.double(0.0315),
    kappa = cms.double(9),
    outlierDeltaFactor = cms.double(2.0)
)

hltHgcalSoARecHitsLayerClustersProducerHSciSerialSync = makeSerialClone(hltHgcalSoARecHitsLayerClustersProducerHSci,
                                                                    hgcalRecHitsSoA = "hltHgcalSoARecHitsProducerHSciSerialSync"
)
