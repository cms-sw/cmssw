import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.EcalRecProducers.ecalWeightUncalibRecHit_cfi import ecalWeightUncalibRecHit
ecalWeightUncalibRecHitSelectedDigis = ecalWeightUncalibRecHit.clone()
ecalWeightUncalibRecHitSelectedDigis.EBdigiCollection = cms.InputTag("selectDigi","selectedEcalEBDigiCollection")
ecalWeightUncalibRecHitSelectedDigis.EEdigiCollection = cms.InputTag("selectDigi","selectedEcalEEDigiCollection")

from RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi import ecalRecHit
ecalWeightRecHitSelectedDigis = ecalRecHit.clone()
ecalWeightRecHitSelectedDigis.EEuncalibRecHitCollection = cms.InputTag("ecalWeightUncalibRecHitSelectedDigis","EcalUncalibRecHitsEE")
ecalWeightRecHitSelectedDigis.EBuncalibRecHitCollection = cms.InputTag("ecalWeightUncalibRecHitSelectedDigis","EcalUncalibRecHitsEB")
ecalWeightRecHitSelectedDigis.recoverEBFE = cms.bool(False)
ecalWeightRecHitSelectedDigis.recoverEEFE = cms.bool(False)
ecalWeightRecHitSelectedDigis.killDeadChannels = cms.bool(False)

ecalWeightLocalRecoFromSelectedDigis =cms.Sequence(ecalWeightUncalibRecHitSelectedDigis*
                                                   ecalWeightRecHitSelectedDigis)

