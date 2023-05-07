import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFClusterProducer.particleFlowRecHitHBHE_cfi import _thresholdsHBphase1, _thresholdsHEphase1, _thresholdsHBphase1_2022_rereco

egammaHBHERecHit = cms.PSet(
    hbheRecHits = cms.InputTag('hbhereco'),
    recHitEThresholdHB = _thresholdsHBphase1,
    recHitEThresholdHE = _thresholdsHEphase1,
    maxHcalRecHitSeverity = cms.int32(9),
)

egammaHBHERecHit_2022_rereco = egammaHBHERecHit.clone(
    recHitEThresholdHB = _thresholdsHBphase1_2022_rereco
)

from Configuration.Eras.Modifier_run3_egamma_2022_rereco_cff import run3_egamma_2022_rereco
run3_egamma_2022_rereco.toReplaceWith(egammaHBHERecHit,egammaHBHERecHit_2022_rereco)
