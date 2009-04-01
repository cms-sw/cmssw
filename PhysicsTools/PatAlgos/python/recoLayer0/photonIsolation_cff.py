import FWCore.ParameterSet.Config as cms

### Compute isolation values, using POG modules
from RecoEgamma.EgammaIsolationAlgos.gamIsoDeposits_cff import *
from RecoEgamma.EgammaIsolationAlgos.gamIsoFromDepsModules_cff  import *

# sequence to run on AOD 
patPhotonIsolation = cms.Sequence(
    gamIsoDepositTk * gamIsoFromDepsTk +
    gamIsoDepositEcalFromHits * gamIsoFromDepsEcalFromHits +
    gamIsoDepositHcalFromTowers * gamIsoFromDepsHcalFromTowers
)
