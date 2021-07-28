import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.egammaHBHERecHitThreshold_cff import egammaHBHERecHit

import RecoEgamma.EgammaElectronProducers.gsfElectronProducerDefault_cfi as _gsfProd
gsfElectronProducer = _gsfProd.gsfElectronProducerDefault.clone(
    hbheRecHits = egammaHBHERecHit.hbheRecHits,
    recHitEThresholdHB = egammaHBHERecHit.recHitEThresholdHB,
    recHitEThresholdHE = egammaHBHERecHit.recHitEThresholdHE,
    maxHcalRecHitSeverity = egammaHBHERecHit.maxHcalRecHitSeverity
)
