import FWCore.ParameterSet.Config as cms

import RecoEcal.EgammaClusterProducers.interestingDetIdCollectionProducer_cfi
pfPhotonInterestingEcalDetIdEB = RecoEcal.EgammaClusterProducers.interestingDetIdCollectionProducer_cfi.interestingDetIdCollectionProducer.clone(
    basicClustersLabel = "pfPhotonTranslator:pfphot",
    recHitsLabel       = "ecalRecHit:EcalRecHitsEB"
    )

pfPhotonInterestingEcalDetIdEE = RecoEcal.EgammaClusterProducers.interestingDetIdCollectionProducer_cfi.interestingDetIdCollectionProducer.clone(
    basicClustersLabel = "pfPhotonTranslator:pfphot",
    recHitsLabel       = "ecalRecHit:EcalRecHitsEE"
    )
