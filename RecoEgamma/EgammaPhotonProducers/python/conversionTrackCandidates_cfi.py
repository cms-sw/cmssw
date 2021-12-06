import FWCore.ParameterSet.Config as cms
from RecoEgamma.EgammaIsolationAlgos.egammaHBHERecHitThreshold_cff import egammaHBHERecHit

import RecoEgamma.EgammaPhotonProducers.conversionTrackCandidatesDefault_cfi as _convTrkCand
conversionTrackCandidates = _convTrkCand.conversionTrackCandidatesDefault.clone(
    hbheRecHits = egammaHBHERecHit.hbheRecHits,
    recHitEThresholdHB = egammaHBHERecHit.recHitEThresholdHB,
    recHitEThresholdHE = egammaHBHERecHit.recHitEThresholdHE,
    maxHcalRecHitSeverity = egammaHBHERecHit.maxHcalRecHitSeverity
)
