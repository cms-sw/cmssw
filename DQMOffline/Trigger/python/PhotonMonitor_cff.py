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


exoHLTPhotonmonitoring = cms.Sequence(
    SinglePhoton300_monitoring
    + SinglePhoton175_monitoring
)





