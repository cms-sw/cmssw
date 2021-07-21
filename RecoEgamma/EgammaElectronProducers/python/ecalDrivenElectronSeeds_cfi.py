import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.egammaHBHERecHitThreshold_cff import egammaHBHERecHit

import RecoEgamma.EgammaElectronProducers.ecalDrivenElectronSeedsDefault_cfi as _ecalSeeds
ecalDrivenElectronSeeds = _ecalSeeds.ecalDrivenElectronSeedsDefault.clone(
    hbheRecHits = egammaHBHERecHit.hbheRecHits,
    recHitEThresholdHB = egammaHBHERecHit.recHitEThresholdHB,
    recHitEThresholdHE = egammaHBHERecHit.recHitEThresholdHE,
    maxHcalRecHitSeverity = egammaHBHERecHit.maxHcalRecHitSeverity
)
