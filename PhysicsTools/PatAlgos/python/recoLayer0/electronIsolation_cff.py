import FWCore.ParameterSet.Config as cms

## compute isolation, using POG modules
from RecoEgamma.EgammaIsolationAlgos.eleIsoDeposits_cff import eleIsoDepositTk, eleIsoDepositEcalFromHits, eleIsoDepositHcalFromTowers
from RecoEgamma.EgammaIsolationAlgos.eleIsoFromDeposits_cff import eleIsoFromDepsTk, eleIsoFromDepsEcalFromHitsByCrystal, eleIsoFromDepsHcalFromTowers


eleIsoDepositEcalFromHits.ExtractorPSet.barrelEcalHits = cms.InputTag("reducedEcalRecHitsEB")
eleIsoDepositEcalFromHits.ExtractorPSet.endcapEcalHits = cms.InputTag("reducedEcalRecHitsEE")

## sequence to run on AOD 
patElectronIsolation = cms.Sequence(
    eleIsoDepositTk * eleIsoFromDepsTk +
    eleIsoDepositEcalFromHits * eleIsoFromDepsEcalFromHitsByCrystal +
    eleIsoDepositHcalFromTowers * eleIsoFromDepsHcalFromTowers
)

