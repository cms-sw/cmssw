import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.AlpakaCore.functions import makeSerialClone

hltHgcalSoARecHitsLayerClustersProducerHSi = cms.EDProducer("HGCalCLUEsteringLayerClustersProducer@alpaka",
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    ),
    hgcalRecHitsSoA = cms.InputTag("hltHgcalSoARecHitsProducerHSi"),
    detector = cms.string('FH'),
    deltac = cms.float(1.3),
    kappa = cms.float(9),
    outlierDeltaFactor = cms.float(2.0)
)

hltHgcalSoARecHitsLayerClustersProducerHSiSerialSync = makeSerialClone(hltHgcalSoARecHitsLayerClustersProducerHSi,
                                                                    hgcalRecHitsSoA = "hltHgcalSoARecHitsProducerHSiSerialSync"
)
