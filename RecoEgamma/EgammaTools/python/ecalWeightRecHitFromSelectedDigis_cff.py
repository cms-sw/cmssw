import FWCore.ParameterSet.Config as cms

import RecoLocalCalo.EcalRecProducers.ecalWeightUncalibRecHit_cfi

ecalWeightUncalibRecHitSelectedDigis = RecoLocalCalo.EcalRecProducers.ecalWeightUncalibRecHit_cfi.ecalWeightUncalibRecHit.clone()
ecalWeightUncalibRecHitSelectedDigis.EBdigiCollection = cms.InputTag("selectDigi","selectedEcalEBDigiCollection")
ecalWeightUncalibRecHitSelectedDigis.EEdigiCollection = cms.InputTag("selectDigi","selectedEcalEEDigiCollection")

import RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi
ecalWeightRecHitSelectedDigis = RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi.ecalRecHit.clone()
ecalWeightRecHitSelectedDigis.EEuncalibRecHitCollection = cms.InputTag("ecalWeightUncalibRecHitSelectedDigis","EcalUncalibRecHitsEE")
ecalWeightRecHitSelectedDigis.EBuncalibRecHitCollection = cms.InputTag("ecalWeightUncalibRecHitSelectedDigis","EcalUncalibRecHitsEB")
ecalWeightRecHitSelectedDigis.recoverEBFE = cms.bool(False)
ecalWeightRecHitSelectedDigis.recoverEEFE = cms.bool(False)
ecalWeightRecHitSelectedDigis.killDeadChannels = cms.bool(False)

ecalWeightLocalRecoFromSelectedDigis =cms.Sequence(ecalWeightUncalibRecHitSelectedDigis*
                                                   ecalWeightRecHitSelectedDigis)

