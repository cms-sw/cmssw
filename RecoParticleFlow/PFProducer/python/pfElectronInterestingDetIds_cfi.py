import FWCore.ParameterSet.Config as cms

import RecoEcal.EgammaClusterProducers.interestingDetIdCollectionProducer_cfi

pfElectronInterestingEcalDetIdEB = RecoEcal.EgammaClusterProducers.interestingDetIdCollectionProducer_cfi.interestingDetIdCollectionProducer.clone(
    basicClustersLabel = "pfElectronTranslator:pf",
    recHitsLabel       = "ecalRecHit:EcalRecHitsEB"
    )
pfElectronInterestingEcalDetIdEE = RecoEcal.EgammaClusterProducers.interestingDetIdCollectionProducer_cfi.interestingDetIdCollectionProducer.clone(
    basicClustersLabel = "pfElectronTranslator:pf",
    recHitsLabel       = "ecalRecHit:EcalRecHitsEE"
    )
