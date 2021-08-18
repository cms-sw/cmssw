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
    pfIsolCfg = cms.PSet(
        chargedHadronIso = cms.InputTag(""),
        neutralHadronIso = cms.InputTag(""),
        photonIso = cms.InputTag(""),
        chargedHadronWorstVtxIso = cms.InputTag(""),
        chargedHadronWorstVtxGeomVetoIso = cms.InputTag(""),
        chargedHadronPFPVIso = cms.InputTag(""),
    ),
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
    hbheRecHits = ""
)
run2_miniAOD_80XLegacy.toModify(
    ootPhotons, 
    barrelEcalHits = "reducedEcalRecHitsEB",
    endcapEcalHits = "reducedEcalRecHitsEE",
    preshowerHits = "reducedEcalRecHitsES",
    hbheRecHits = "",
    pfECALClusIsolation = None,
    pfHCALClusIsolation = None
)
run2_miniAOD_80XLegacy.toModify(
    ootPhotonsTmp.isolationSumsCalculatorSet, 
    barrelEcalRecHitCollection = "reducedEcalRecHitsEB",
    endcapEcalRecHitCollection = "reducedEcalRecHitsEE",
    HBHERecHitCollection = ""
)
run2_miniAOD_80XLegacy.toModify(
    ootPhotonsTmp.mipVariableSet,
    barrelEcalRecHitCollection = "reducedEcalRecHitsEB",
    endcapEcalRecHitCollection = "reducedEcalRecHitsEE",
)
