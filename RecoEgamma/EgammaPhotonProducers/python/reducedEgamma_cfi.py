import FWCore.ParameterSet.Config as cms

reducedEgamma = cms.EDProducer("ReducedEGProducer",
  keepPhotons = cms.string("pt > 14 && hadTowOverEm()<0.15"),
  relinkPhotons = cms.string("(r9()>0.8 || chargedHadronIso()<20 || chargedHadronIso()<0.3*pt())"),
  keepGsfElectrons = cms.string(""),
  relinkGsfElectrons = cms.string("pt>5"),
  photons = cms.InputTag("gedPhotons"),
  gsfElectrons = cms.InputTag("gedGsfElectrons"),
  conversions = cms.InputTag("allConversions"),
  singleConversions = cms.InputTag("gedPhotonCore"),
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
)


