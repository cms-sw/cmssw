import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFClusterProducer.particleFlowRecHitHBHE_cfi import _thresholdsHBphase1, _thresholdsHEphase1, _thresholdsHBphase1_2023

egammaHBHERecHit = cms.PSet(
    hbheRecHits = cms.InputTag('hbhereco'),
    recHitEThresholdHB = _thresholdsHBphase1,
    recHitEThresholdHE = _thresholdsHEphase1,
    maxHcalRecHitSeverity = cms.int32(9),
    usePFThresholdsFromDB = cms.bool(False)
)

from Configuration.Eras.Modifier_hcalPfCutsFromDB_cff import hcalPfCutsFromDB
hcalPfCutsFromDB.toModify(egammaHBHERecHit,
                   usePFThresholdsFromDB = True)


