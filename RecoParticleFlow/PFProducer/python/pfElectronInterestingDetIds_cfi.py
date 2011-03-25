import FWCore.ParameterSet.Config as cms

import RecoEcal.EgammaClusterProducers.interestingDetIdCollectionProducer_cfi

pfElectronInterestingEcalDetIdEB = RecoEcal.EgammaClusterProducers.interestingDetIdCollectionProducer_cfi.interestingDetIdCollectionProducer.clone(
    basicClustersLabel = cms.InputTag("pfElectronTranslator","pf"),
    recHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB")
    )
pfElectronInterestingEcalDetIdEE = RecoEcal.EgammaClusterProducers.interestingDetIdCollectionProducer_cfi.interestingDetIdCollectionProducer.clone(
    basicClustersLabel = cms.InputTag("pfElectronTranslator","pf"),
    recHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE")
    )
