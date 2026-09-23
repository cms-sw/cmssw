import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.AlpakaCore.functions import makeSerialClone

hltHgcalSoARecHitsLayerClustersProducer = cms.EDProducer("HGCalSoARecHitsLayerClustersProducer@alpaka",
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    ),
    hgcalRecHitsSoA = cms.InputTag("hltHgcalSoARecHitsProducer"),
    deltac = cms.float(1.3),
    kappa = cms.float(9),
    outlierDeltaFactor = cms.float(2.0)
)

hltHgcalSoARecHitsLayerClustersProducerSerialSync = makeSerialClone(hltHgcalSoARecHitsLayerClustersProducer,
                                                                    hgcalRecHitsSoA = "hltHgcalSoARecHitsProducerSerialSync"
)
