import FWCore.ParameterSet.Config as cms

import RecoEcal.EgammaClusterProducers.interestingDetIdCollectionProducer_cfi
pfPhotonInterestingEcalDetIdEB = RecoEcal.EgammaClusterProducers.interestingDetIdCollectionProducer_cfi.interestingDetIdCollectionProducer.clone(
    basicClustersLabel = cms.InputTag("pfPhotonTranslator","pfphot"),
    recHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB")
    )

pfPhotonInterestingEcalDetIdEE = RecoEcal.EgammaClusterProducers.interestingDetIdCollectionProducer_cfi.interestingDetIdCollectionProducer.clone(
    basicClustersLabel = cms.InputTag("pfPhotonTranslator","pfphot"),
    recHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE")
    )
