import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.AlpakaCore.functions import makeSerialClone

hltHgcalSoALayerClustersProducerHSi = cms.EDProducer("HGCalSoALayerClustersProducer@alpaka",
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    ),
    hgcalRecHitsLayerClustersSoA = cms.InputTag("hltHgcalSoARecHitsLayerClustersProducerHSi"),
    hgcalRecHitsSoA = cms.InputTag("hltHgcalSoARecHitsProducerHSi"),
    detector = cms.string('FH'),
    positionDeltaRho2 = cms.float(1.69),
    thresholdW0 = cms.float(2.9)
)

hltHgcalSoALayerClustersProducerHSiSerialSync = makeSerialClone(hltHgcalSoALayerClustersProducerHSi,
                                                             hgcalRecHitsLayerClustersSoA = "hltHgcalSoARecHitsLayerClustersProducerHSiSerialSync",
                                                             hgcalRecHitsSoA = "hltHgcalSoARecHitsProducerHSiSerialSync"
)
