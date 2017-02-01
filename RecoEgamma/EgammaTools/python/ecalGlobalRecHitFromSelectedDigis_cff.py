import FWCore.ParameterSet.Config as cms

import RecoLocalCalo.EcalRecProducers.ecalGlobalUncalibRecHit_cfi

ecalGlobalUncalibRecHitSelectedDigis = RecoLocalCalo.EcalRecProducers.ecalGlobalUncalibRecHit_cfi.ecalGlobalUncalibRecHit.clone()
ecalGlobalUncalibRecHitSelectedDigis.EBdigiCollection = cms.InputTag("selectDigi","selectedEcalEBDigiCollection")
ecalGlobalUncalibRecHitSelectedDigis.EEdigiCollection = cms.InputTag("selectDigi","selectedEcalEEDigiCollection")

import RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi
ecalGlobalRecHitSelectedDigis = RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi.ecalRecHit.clone()
ecalGlobalRecHitSelectedDigis.EEuncalibRecHitCollection = cms.InputTag("ecalGlobalUncalibRecHitSelectedDigis","EcalUncalibRecHitsEE")
ecalGlobalRecHitSelectedDigis.EBuncalibRecHitCollection = cms.InputTag("ecalGlobalUncalibRecHitSelectedDigis","EcalUncalibRecHitsEB")
ecalGlobalRecHitSelectedDigis.recoverEBFE = cms.bool(False)
ecalGlobalRecHitSelectedDigis.recoverEEFE = cms.bool(False)
ecalGlobalRecHitSelectedDigis.killDeadChannels = cms.bool(False)

ecalGlobalLocalRecoFromSelectedDigis =cms.Sequence(ecalGlobalUncalibRecHitSelectedDigis*
                                                   ecalGlobalRecHitSelectedDigis)

