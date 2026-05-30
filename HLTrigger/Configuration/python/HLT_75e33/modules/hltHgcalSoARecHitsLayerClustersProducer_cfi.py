import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.AlpakaCore.functions import makeSerialClone

hltHgcalSoARecHitsLayerClustersProducer = cms.EDProducer("HGCalSoARecHitsLayerClustersProducer@alpaka",
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    ),
    hgcalRecHitsSoA = cms.InputTag("hltHgcalSoARecHitsProducer"),
    deltac = cms.double(1.3),
    kappa = cms.double(9),
    outlierDeltaFactor = cms.double(2.0)
)

hltHgcalSoARecHitsLayerClustersProducerSerialSync = makeSerialClone(hltHgcalSoARecHitsLayerClustersProducer,
                                                                    hgcalRecHitsSoA = "hltHgcalSoARecHitsProducerSerialSync"
)
