import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFClusterProducer.particleFlowRecHitHBHE_cfi import _thresholdsHBphase1, _thresholdsHEphase1

egammaHBHERecHit = cms.PSet(
    hbheRecHits = cms.InputTag('hbhereco'),
    recHitEThresholdHB = _thresholdsHBphase1,
    recHitEThresholdHE = _thresholdsHEphase1,
    maxHcalRecHitSeverity = cms.int32(9),
)
