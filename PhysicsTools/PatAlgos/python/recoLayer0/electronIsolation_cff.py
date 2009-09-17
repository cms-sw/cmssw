import FWCore.ParameterSet.Config as cms

## compute isolation, using POG modules
from RecoEgamma.EgammaIsolationAlgos.eleIsoDeposits_cff import *
from RecoEgamma.EgammaIsolationAlgos.eleIsoFromDeposits_cff  import *

eleIsoDepositEcalFromHits.ExtractorPSet.barrelEcalHits = cms.InputTag("reducedEcalRecHitsEB", "", "RECO")
eleIsoDepositEcalFromHits.ExtractorPSet.endcapEcalHits = cms.InputTag("reducedEcalRecHitsEE", "", "RECO")

## sequence to run on AOD 
patElectronIsolation = cms.Sequence(
    eleIsoDepositTk * eleIsoFromDepsTk +
    eleIsoDepositEcalFromHits * eleIsoFromDepsEcalFromHits +
    eleIsoDepositHcalFromTowers * eleIsoFromDepsHcalFromTowers
)

