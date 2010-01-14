import FWCore.ParameterSet.Config as cms

### Compute isolation values, using POG modules
from RecoEgamma.EgammaIsolationAlgos.gamIsoDeposits_cff import gamIsoDepositTk, gamIsoDepositEcalFromHits, gamIsoDepositHcalFromTowers
from RecoEgamma.EgammaIsolationAlgos.gamIsoFromDepsModules_cff import gamIsoFromDepsTk, gamIsoFromDepsEcalFromHits, gamIsoFromDepsHcalFromTowers


gamIsoDepositEcalFromHits.ExtractorPSet.barrelEcalHits = cms.InputTag("reducedEcalRecHitsEB")
gamIsoDepositEcalFromHits.ExtractorPSet.endcapEcalHits = cms.InputTag("reducedEcalRecHitsEE")

# sequence to run on AOD 
patPhotonIsolation = cms.Sequence(
    gamIsoDepositTk * gamIsoFromDepsTk +
    gamIsoDepositEcalFromHits * gamIsoFromDepsEcalFromHits +
    gamIsoDepositHcalFromTowers * gamIsoFromDepsHcalFromTowers
)
