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



