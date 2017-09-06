import FWCore.ParameterSet.Config as cms

reducedEgamma = cms.EDProducer("ReducedEGProducer",
  keepPhotons = cms.string("hadTowOverEm()<0.15 && pt>10 && (pt>14 || chargedHadronIso()<10)"), #keep in output
  slimRelinkPhotons = cms.string("hadTowOverEm()<0.15 && pt>10 && (pt>14 || chargedHadronIso()<10)"), #keep only slimmed SuperCluster plus seed cluster
  relinkPhotons = cms.string("(r9()>0.8 || chargedHadronIso()<20 || chargedHadronIso()<0.3*pt())"), #keep all associated clusters/rechits/conversions
  keepOOTPhotons = cms.string("pt>10"), #keep in output
  slimRelinkOOTPhotons = cms.string("pt>10"), #keep only slimmed SuperCluster plus seed cluster
  relinkOOTPhotons = cms.string("(r9()>0.8)"), #keep all associated clusters/rechits/conversions
  keepGsfElectrons = cms.string(""), #keep in output
  slimRelinkGsfElectrons = cms.string(""), #keep only slimmed SuperCluster plus seed cluster
  relinkGsfElectrons = cms.string("pt>5"), #keep all associated clusters/rechits/conversions
  photons = cms.InputTag("gedPhotons"),
  ootPhotons = cms.InputTag("ootPhotons"),
  gsfElectrons = cms.InputTag("gedGsfElectrons"),
  conversions = cms.InputTag("allConversions"),
  gsfTracks = cms.InputTag("electronGsfTracks"),
  singleConversions = cms.InputTag("particleFlowEGamma"),
  barrelEcalHits = cms.InputTag("reducedEcalRecHitsEB"),
  endcapEcalHits = cms.InputTag("reducedEcalRecHitsEE"),
  preshowerEcalHits = cms.InputTag("reducedEcalRecHitsES"),
  photonsPFValMap = cms.InputTag("particleBasedIsolation","gedPhotons"),
  gsfElectronsPFValMap = cms.InputTag("particleBasedIsolation","gedGsfElectrons"),
  photonIDSources = cms.VInputTag(
    cms.InputTag("PhotonIDProdGED","PhotonCutBasedIDLoose"),
    cms.InputTag("PhotonIDProdGED","PhotonCutBasedIDLooseEM"),    
    cms.InputTag("PhotonIDProdGED","PhotonCutBasedIDTight")
  ),
  photonIDOutput = cms.vstring(
    "PhotonCutBasedIDLoose",
    "PhotonCutBasedIDLooseEM",
    "PhotonCutBasedIDTight",
  ),
  gsfElectronIDSources = cms.VInputTag(
    cms.InputTag("eidLoose"),
    cms.InputTag("eidRobustHighEnergy"),
    cms.InputTag("eidRobustLoose"),
    cms.InputTag("eidRobustTight"),
    cms.InputTag("eidTight"),
  ),
  gsfElectronIDOutput = cms.vstring(
    "eidLoose",
    "eidRobustHighEnergy",
    "eidRobustLoose",
    "eidRobustTight",
    "eidTight",
    ),
  photonPFClusterIsoSources = cms.VInputTag(
        cms.InputTag("photonEcalPFClusterIsolationProducer"),
        cms.InputTag("photonHcalPFClusterIsolationProducer"),
  ),
  photonPFClusterIsoOutput = cms.vstring(
        "phoEcalPFClusIso",
        "phoHcalPFClusIso",
  ),
  ootPhotonPFClusterIsoSources = cms.VInputTag(
        cms.InputTag("ootPhotonEcalPFClusterIsolationProducer"),
        cms.InputTag("ootPhotonHcalPFClusterIsolationProducer"),
  ),
  ootPhotonPFClusterIsoOutput = cms.vstring(
        "ootPhoEcalPFClusIso",
        "ootPhoHcalPFClusIso",
  ),
  gsfElectronPFClusterIsoSources = cms.VInputTag(
        cms.InputTag("electronEcalPFClusterIsolationProducer"),
        cms.InputTag("electronHcalPFClusterIsolationProducer"),
  ),
  gsfElectronPFClusterIsoOutput = cms.vstring(
        "eleEcalPFClusIso",
        "eleHcalPFClusIso",
  ),
)

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify(reducedEgamma, 
        preshowerEcalHits = cms.InputTag(""),
)

from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy
run2_miniAOD_80XLegacy.toModify(
    reducedEgamma, 
    ootPhotonPFClusterIsoSources = [ "ootPhotonEcalPFClusterIsolationProducer" ],
    ootPhotonPFClusterIsoOutput = [ "ootPhoEcalPFClusIso" ]
)
