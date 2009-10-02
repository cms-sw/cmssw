import FWCore.ParameterSet.Config as cms

### Compute isolation values, using POG modules
from RecoEgamma.EgammaIsolationAlgos.gamIsoDeposits_cff import *
from RecoEgamma.EgammaIsolationAlgos.gamIsoFromDepsModules_cff  import *

gamIsoDepositEcalFromHits.ExtractorPSet.barrelEcalHits = cms.InputTag("reducedEcalRecHitsEB", "", "RECO")
gamIsoDepositEcalFromHits.ExtractorPSet.endcapEcalHits = cms.InputTag("reducedEcalRecHitsEE", "", "RECO")


# sequence to run on AOD 
patPhotonIsolation = cms.Sequence(
    gamIsoDepositTk * gamIsoFromDepsTk +
    gamIsoDepositEcalFromHits * gamIsoFromDepsEcalFromHits +
    gamIsoDepositHcalFromTowers * gamIsoFromDepsHcalFromTowers
)
