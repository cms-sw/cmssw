import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.AlpakaCore.functions import makeSerialClone

hltHgcalSoALayerClustersProducer = cms.EDProducer("HGCalSoALayerClustersProducer@alpaka",
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    ),
    hgcalRecHitsLayerClustersSoA = cms.InputTag("hltHgcalSoARecHitsLayerClustersProducer"),
    hgcalRecHitsSoA = cms.InputTag("hltHgcalSoARecHitsProducer"),
    positionDeltaRho2 = cms.float(1.69),
    thresholdW0 = cms.float(2.9)
)

hltHgcalSoALayerClustersProducerSerialSync = makeSerialClone(hltHgcalSoALayerClustersProducer,
                                                             #feed the upstream serial modules in
                                                             hgcalRecHitsLayerClustersSoA = "hltHgcalSoARecHitsLayerClustersProducerSerialSync",
                                                             hgcalRecHitsSoA = "hltHgcalSoARecHitsProducerSerialSync"
)
