import FWCore.ParameterSet.Config as cms

### Compute isolation values, using POG modules
from RecoEgamma.EgammaIsolationAlgos.eleIsoDeposits_cff import *
from RecoEgamma.EgammaIsolationAlgos.eleIsoFromDepsModules_cff  import *

# sequence to run on AOD 
patElectronIsolation = cms.Sequence(
    eleIsoDepositTk * eleIsoFromDepsTk +
    eleIsoDepositEcalFromHits * eleIsoFromDepsEcalFromHits +
    eleIsoDepositHcalFromTowers * eleIsoFromDepsHcalFromTowers
)

