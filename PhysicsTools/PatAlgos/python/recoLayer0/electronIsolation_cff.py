import FWCore.ParameterSet.Config as cms

## compute isolation, using POG modules
from RecoEgamma.EgammaIsolationAlgos.eleIsoDeposits_cff import eleIsoDepositTk, eleIsoDepositEcalFromHits, eleIsoDepositHcalFromTowers
from RecoEgamma.EgammaIsolationAlgos.eleIsoFromDeposits_cff import eleIsoFromDepsTk, eleIsoFromDepsEcalFromHitsByCrystal, eleIsoFromDepsHcalFromTowers
 

eleIsoDepositEcalFromHits.ExtractorPSet.barrelEcalHits = cms.InputTag("reducedEcalRecHitsEB")
eleIsoDepositEcalFromHits.ExtractorPSet.endcapEcalHits = cms.InputTag("reducedEcalRecHitsEE")

## sequence to run on AOD 
patElectronTrackIsolation = cms.Sequence(
    eleIsoDepositTk * eleIsoFromDepsTk
)

patElectronEcalIsolation = cms.Sequence(
    eleIsoDepositEcalFromHits * eleIsoFromDepsEcalFromHitsByCrystal
)

patElectronHcalIsolation = cms.Sequence(
    eleIsoDepositHcalFromTowers * eleIsoFromDepsHcalFromTowers
)

patElectronIsolation = cms.Sequence(
    patElectronTrackIsolation +
    patElectronEcalIsolation  +
    patElectronHcalIsolation
)
