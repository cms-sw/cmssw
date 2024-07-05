import FWCore.ParameterSet.Config as cms

hltHgcalSoALayerClustersProducer = cms.EDProducer("HGCalSoALayerClustersProducer@alpaka",
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    ),
    hgcalRecHitsLayerClustersSoA = cms.InputTag("hltHgcalSoARecHitsLayerClustersProducer"),
    hgcalRecHitsSoA = cms.InputTag("hltHgcalSoARecHitsProducer"),
    positionDeltaRho2 = cms.double(1.69),
    thresholdW0 = cms.double(2.9)
)
