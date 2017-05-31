import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.PhotonMonitor_cfi import hltPhotonmonitoring

#HLT_SinglePhoton200_IDTight
SinglePhoton300_monitoring = hltPhotonmonitoring.clone()
SinglePhoton300_monitoring.FolderName = cms.string('HLT/Photon/Photon300/')
SinglePhoton300_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Photon300_NoHE_v")


# HLT_SinglePhoton175_IDTight
SinglePhoton175_monitoring = hltPhotonmonitoring.clone()
SinglePhoton175_monitoring.FolderName = cms.string('HLT/Photon/Photon175/')
SinglePhoton175_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Photon175_v*")

SinglePhoton50_R9Id90_HE10_IsoM_monitoring = hltPhotonmonitoring.clone()
SinglePhoton50_R9Id90_HE10_IsoM_monitoring.FolderName = cms.string('HLT/Photon/Photon50_R9Id90_HE10_IsoM/')
SinglePhoton50_R9Id90_HE10_IsoM_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Photon50_R9Id90_HE10_IsoM_v*")


SinglePhoton75_R9Id90_HE10_IsoM_monitoring = hltPhotonmonitoring.clone()
SinglePhoton75_R9Id90_HE10_IsoM_monitoring.FolderName = cms.string('HLT/Photon/Photon75_R9Id90_HE10_IsoM/')
SinglePhoton75_R9Id90_HE10_IsoM_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Photon75_R9Id90_HE10_IsoM_v*")


SinglePhoton90_R9Id90_HE10_IsoM_monitoring = hltPhotonmonitoring.clone()
SinglePhoton90_R9Id90_HE10_IsoM_monitoring.FolderName = cms.string('HLT/Photon/Photon90_R9Id90_HE10_IsoM/')
SinglePhoton90_R9Id90_HE10_IsoM_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Photon90_R9Id90_HE10_IsoM_v*")

SinglePhoton120_R9Id90_HE10_IsoM_monitoring = hltPhotonmonitoring.clone()
SinglePhoton120_R9Id90_HE10_IsoM_monitoring.FolderName = cms.string('HLT/Photon/Photon120_R9Id90_HE10_IsoM/')
SinglePhoton120_R9Id90_HE10_IsoM_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Photon120_R9Id90_HE10_IsoM_v*")

SinglePhoton165_R9Id90_HE10_IsoM_monitoring = hltPhotonmonitoring.clone()
SinglePhoton165_R9Id90_HE10_IsoM_monitoring.FolderName = cms.string('HLT/Photon/Photon165_R9Id90_HE10_IsoM/')
SinglePhoton165_R9Id90_HE10_IsoM_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Photon165_R9Id90_HE10_IsoM_v*")


Photon60_monitoring = hltPhotonmonitoring.clone()
Photon60_monitoring.FolderName = cms.string('HLT/Photon/Photon60/')
Photon60_monitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring()
Photon60_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Photon60_R9Id90_CaloIdL_IsoL_v*")
Photon60_monitoring.photonSelection = cms.string("pt > 20 && r9() < 0.1 && ((eta<1.4442 && hadTowOverEm<0.0597 && full5x5_sigmaIetaIeta()<0.01031 && chargedHadronIso<1.295) || (eta<2.5 && eta>1.566 && hadTowOverEm<0.0481 && full5x5_sigmaIetaIeta()<0.03013 && chargedHadronIso<1.011))")

Photon60_DisplacedIdL_monitoring = Photon60_monitoring.clone()
Photon60_DisplacedIdL_monitoring.FolderName = cms.string('HLT/Photon/Photon60_DisplacedIdL/')
Photon60_DisplacedIdL_monitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Photon60_R9Id90_CaloIdL_IsoL_v*")
Photon60_DisplacedIdL_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Photon60_R9Id90_CaloIdL_IsoL_DisplacedIdL_v*")

Photon60_DisplacedIdL_PFJet350MinPFJet15_monitoring = Photon60_DisplacedIdL_monitoring.clone()
Photon60_DisplacedIdL_PFJet350MinPFJet15_monitoring.denGenericTriggerEventPSet.andOrHlt = cms.bool(False)
Photon60_DisplacedIdL_PFJet350MinPFJet15_monitoring.FolderName = cms.string('HLT/Photon/Photon60_DisplacedIdL_PFJet350MinPFJet15/')
Photon60_DisplacedIdL_PFJet350MinPFJet15_monitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Photon60_R9Id90_CaloIdL_IsoL_v*","HLT_PFHT350MinPFJet15_v*")
Photon60_DisplacedIdL_PFJet350MinPFJet15_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Photon60_R9Id90_CaloIdL_IsoL_DisplacedIdL_PFHT350MinPFJet15_v*")


exoHLTPhotonmonitoring = cms.Sequence(
    SinglePhoton300_monitoring
    + SinglePhoton175_monitoring
    + Photon60_monitoring
    + Photon60_DisplacedIdL_monitoring
    + Photon60_DisplacedIdL_PFJet350MinPFJet15_monitoring
    + SinglePhoton50_R9Id90_HE10_IsoM_monitoring
    + SinglePhoton75_R9Id90_HE10_IsoM_monitoring
    + SinglePhoton90_R9Id90_HE10_IsoM_monitoring
    + SinglePhoton120_R9Id90_HE10_IsoM_monitoring
    + SinglePhoton165_R9Id90_HE10_IsoM_monitoring
    
)





