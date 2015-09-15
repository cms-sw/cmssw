import FWCore.ParameterSet.Config as cms

reducedEgamma = cms.EDProducer("ReducedEGProducer",
  keepPhotons = cms.string("obj.hadTowOverEm()<0.15 && obj.pt()>10 && (obj.pt()>14 || obj.chargedHadronIso()<10)"), #keep in output
  slimRelinkPhotons = cms.string("obj.hadTowOverEm()<0.15 && obj.pt()>10 && (obj.pt()>14 || obj.chargedHadronIso()<10)"), #keep only slimmed SuperCluster plus seed cluster
  relinkPhotons = cms.string("(obj.r9()>0.8 || obj.chargedHadronIso()<20 || obj.chargedHadronIso()<0.3*obj.pt())"), #keep all associated clusters/rechits/conversions
  keepGsfElectrons = cms.string(""), #keep in output
  slimRelinkGsfElectrons = cms.string(""), #keep only slimmed SuperCluster plus seed cluster
  relinkGsfElectrons = cms.string("obj.pt()>5"), #keep all associated clusters/rechits/conversions
  photons = cms.InputTag("gedPhotons"),
  gsfElectrons = cms.InputTag("gedGsfElectrons"),
  conversions = cms.InputTag("allConversions"),
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
  gsfElectronPFClusterIsoSources = cms.VInputTag(
        cms.InputTag("electronEcalPFClusterIsolationProducer"),
        cms.InputTag("electronHcalPFClusterIsolationProducer"),
  ),
  gsfElectronPFClusterIsoOutput = cms.vstring(
        "eleEcalPFClusIso",
        "eleHcalPFClusIso",
  ),
)
