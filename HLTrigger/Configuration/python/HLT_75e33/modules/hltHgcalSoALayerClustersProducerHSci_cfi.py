import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.AlpakaCore.functions import makeSerialClone

hltHgcalSoALayerClustersProducerHSci = cms.EDProducer("HGCalSoALayerClustersProducer@alpaka",
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    ),
    hgcalRecHitsLayerClustersSoA = cms.InputTag("hltHgcalSoARecHitsLayerClustersProducerHSci"),
    hgcalRecHitsSoA = cms.InputTag("hltHgcalSoARecHitsProducerHSci"),
    detector = cms.string('BH'),
    positionDeltaRho2 = cms.double(1.69),
    thresholdW0 = cms.double(2.9)
)

hltHgcalSoALayerClustersProducerHSciSerialSync = makeSerialClone(hltHgcalSoALayerClustersProducerHSci,
                                                             hgcalRecHitsLayerClustersSoA = "hltHgcalSoARecHitsLayerClustersProducerHSciSerialSync",
                                                             hgcalRecHitsSoA = "hltHgcalSoARecHitsProducerHSciSerialSync"
)
