import FWCore.ParameterSet.Config as cms

hltHgcalSoARecHitsLayerClustersProducer = cms.EDProducer("HGCalSoARecHitsLayerClustersProducer@alpaka",
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')
    ),
    hgcalRecHitsSoA = cms.InputTag("hltHgcalSoARecHitsProducer"),
    deltac = cms.double(1.3),
    kappa = cms.double(9),
    outlierDeltaFactor = cms.double(2.0)
)
