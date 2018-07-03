import FWCore.ParameterSet.Config as cms
from RecoEgamma.EgammaPhotonProducers.gedPhotons_cfi import gedPhotons as _gedPhotons

ootPhotonsTmp = _gedPhotons.clone(
    photonProducer = 'ootPhotonCore',
    candidateP4type = "fromEcalEnergy",
    reconstructionStep = "oot",
    pfEgammaCandidates = "",
    valueMapPhotons = ""
    )
del ootPhotonsTmp.regressionConfig

ootPhotons = _gedPhotons.clone(
    photonProducer = 'ootPhotonsTmp',
    candidateP4type = "fromEcalEnergy",
    reconstructionStep = "ootfinal",
    pfEgammaCandidates = "",
    chargedHadronIsolation = cms.InputTag(""),
    neutralHadronIsolation = cms.InputTag(""),
    photonIsolation = cms.InputTag(""),
    pfECALClusIsolation = cms.InputTag("ootPhotonEcalPFClusterIsolationProducer"),
    pfHCALClusIsolation = cms.InputTag("ootPhotonHcalPFClusterIsolationProducer"),
    valueMapPhotons = ""
    )
del ootPhotons.regressionConfig



from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy

run2_miniAOD_80XLegacy.toModify(
    ootPhotonsTmp, 
    barrelEcalHits = "reducedEcalRecHitsEB",
    endcapEcalHits = "reducedEcalRecHitsEE",
    preshowerHits = "reducedEcalRecHitsES",
    hcalTowers = ""
)
run2_miniAOD_80XLegacy.toModify(
    ootPhotons, 
    barrelEcalHits = "reducedEcalRecHitsEB",
    endcapEcalHits = "reducedEcalRecHitsEE",
    preshowerHits = "reducedEcalRecHitsES",
    hcalTowers = "",
    pfECALClusIsolation = None,
    pfHCALClusIsolation = None
)
run2_miniAOD_80XLegacy.toModify(
    ootPhotonsTmp.isolationSumsCalculatorSet, 
    barrelEcalRecHitCollection = "reducedEcalRecHitsEB",
    endcapEcalRecHitCollection = "reducedEcalRecHitsEE",
    HcalRecHitCollection = ""
)
run2_miniAOD_80XLegacy.toModify(
    ootPhotonsTmp.mipVariableSet,
    barrelEcalRecHitCollection = "reducedEcalRecHitsEB",
    endcapEcalRecHitCollection = "reducedEcalRecHitsEE",
)
