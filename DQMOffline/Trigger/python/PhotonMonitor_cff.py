import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.PhotonMonitor_cfi import hltPhotonmonitoring

# HLT_SinglePhoton200_IDTight
SinglePhoton200_monitoring = hltPhotonmonitoring.clone()
SinglePhoton200_monitoring.FolderName = cms.string('HLT/Photon/Photon200/')
SinglePhoton200_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Photon200_v*")


# HLT_SinglePhoton175_IDTight
SinglePhoton175_monitoring = hltPhotonmonitoring.clone()
SinglePhoton175_monitoring.FolderName = cms.string('HLT/Photon/Photon175/')
SinglePhoton175_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Photon175_v*")


exoHLTPhotonmonitoring = cms.Sequence(
    SinglePhoton200_monitoring
    + SinglePhoton175_monitoring
)


DiphotonMass90_monitoring = hltPhotonmonitoring.clone()
DiphotonMass90_monitoring.FolderName = cms.string('HLT/Photon/diphotonMass90/')
DiphotonMass90_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass90_v*")
DiphotonMass90_monitoring.nphotons = cms.int32(2)
DiphotonMass90_monitoring.photonSelection = cms.string("(pt > 20 && eta<1.4442 && hadTowOverEm<0.12 && full5x5_sigmaIetaIeta()<0.015 && full5x5_r9>.5)||(pt > 20 && eta<2.5 && eta>1.5556 && hadTowOverEm<0.12 && full5x5_sigmaIetaIeta()<0.035 && full5x5_r9>.8)")

DiphotonMass95_monitoring = hltPhotonmonitoring.clone()
DiphotonMass95_monitoring.FolderName = cms.string('HLT/Photon/diphotonMass95/')
DiphotonMass95_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass95_v*")
DiphotonMass95_monitoring.nphotons = cms.int32(2)
DiphotonMass95_monitoring.photonSelection = cms.string("(pt > 20 && eta<1.4442 && hadTowOverEm<0.12 && full5x5_sigmaIetaIeta()<0.015 && full5x5_r9>.5)||(pt > 20 && eta<2.5 && eta>1.5556 && hadTowOverEm<0.12 && full5x5_sigmaIetaIeta()<0.035 && full5x5_r9>.8)")

DiphotonMass55AND_monitoring = hltPhotonmonitoring.clone()
DiphotonMass55AND_monitoring.FolderName = cms.string('HLT/Photon/diphotonMass55AND/')
DiphotonMass55AND_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Diphoton30PV_18PV_R9Id_AND_IsoCaloId_AND_HE_R9Id_PixelVeto_Mass55_v*")
DiphotonMass55AND_monitoring.nphotons = cms.int32(2)
DiphotonMass55AND_monitoring.photonSelection = cms.string("(pt > 20 && eta<1.4442 && hadTowOverEm<0.12 && full5x5_sigmaIetaIeta()<0.015 && full5x5_r9>.5)||(pt > 20 && eta<2.5 && eta>1.5556 && hadTowOverEm<0.12 && full5x5_sigmaIetaIeta()<0.035 && full5x5_r9>.8)")

DiphotonMass55AND_monitoring.histoPSet.massBinning = cms.vdouble(50.,51.,52.,53.,54.,55.,56.,57.,58.,59.,60.,61.,62.,63.,64.,65.,66.,67.,68.,69.,70.,75.,80.,90.,110.,150.)


higgsHLTDiphotonMonitoring = cms.Sequence(
    DiphotonMass90_monitoring
    + DiphotonMass55AND_monitoring
)
