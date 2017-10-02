import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.MuonMonitor_cfi import hltMuonmonitoring


TrkMu12_DoubleTrkMu5NoFiltersNoVtx_monitoring = hltMuonmonitoring.clone()
TrkMu12_DoubleTrkMu5NoFiltersNoVtx_monitoring.FolderName = cms.string('HLT/EXO/TrkMu12_DoubleTrkMu5NoFiltersNoVtx/')
TrkMu12_DoubleTrkMu5NoFiltersNoVtx_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_TrkMu12_DoubleTrkMu5NoFiltersNoVtx_v*")
TrkMu12_DoubleTrkMu5NoFiltersNoVtx_monitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFJet40_v*","HLT_PFJet60_v*","HLT_PFJet80_v*") 

TrkMu16_DoubleTrkMu6NoFiltersNoVtx_monitoring = hltMuonmonitoring.clone()
TrkMu16_DoubleTrkMu6NoFiltersNoVtx_monitoring.FolderName = cms.string('HLT/EXO/TrkMu16_DoubleTrkMu6NoFiltersNoVtx/')
TrkMu16_DoubleTrkMu6NoFiltersNoVtx_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_TrkMu16_DoubleTrkMu6NoFiltersNoVtx_v*")
TrkMu16_DoubleTrkMu6NoFiltersNoVtx_monitoring.denGenericTriggerEventPSet.hltPaths      = cms.vstring("HLT_TrkMu12_DoubleTrkMu5NoFiltersNoVtx_v*") 

TrkMu17_DoubleTrkMu8NoFiltersNoVtx_monitoring = hltMuonmonitoring.clone()
TrkMu17_DoubleTrkMu8NoFiltersNoVtx_monitoring.FolderName = cms.string('HLT/EXO/TrkMu17_DoubleTrkMu8NoFiltersNoVtx/')
TrkMu17_DoubleTrkMu8NoFiltersNoVtx_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_TrkMu17_DoubleTrkMu8NoFiltersNoVtx_v*")
TrkMu17_DoubleTrkMu8NoFiltersNoVtx_monitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_TrkMu12_DoubleTrkMu5NoFiltersNoVtx_v*")



DoubleMu43NoFiltersNoVtx_monitoring = hltMuonmonitoring.clone()
DoubleMu43NoFiltersNoVtx_monitoring.FolderName = cms.string('HLT/EXO/DoubleMu43NoFiltersNoVtx/')
DoubleMu43NoFiltersNoVtx_monitoring.nmuons = cms.uint32(2)
DoubleMu43NoFiltersNoVtx_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleMu43NoFiltersNoVtx_v*")
DoubleMu43NoFiltersNoVtx_monitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET120_PFMHT120_IDTight_v*","HLT_PFMETTypeOne120_PFMHT120_IDTight_v*","HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*")


DoubleMu48NoFiltersNoVtx_monitoring = hltMuonmonitoring.clone()
DoubleMu48NoFiltersNoVtx_monitoring.FolderName = cms.string('HLT/EXO/DoubleMu48NoFiltersNoVtx/')
DoubleMu48NoFiltersNoVtx_monitoring.nmuons = cms.uint32(2)
DoubleMu48NoFiltersNoVtx_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleMu48NoFiltersNoVtx_v*")
DoubleMu48NoFiltersNoVtx_monitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET120_PFMHT120_IDTight_v*","HLT_PFMETTypeOne120_PFMHT120_IDTight_v*","HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*")



Mu43NoFiltersNoVtx_Photon43_CaloIdL_monitoring = hltMuonmonitoring.clone()
Mu43NoFiltersNoVtx_Photon43_CaloIdL_monitoring.FolderName = cms.string('HLT/EXO/Mu43NoFiltersNoVtx_Photon43_CaloIdL/')
Mu43NoFiltersNoVtx_Photon43_CaloIdL_monitoring.nmuons = cms.uint32(1)
Mu43NoFiltersNoVtx_Photon43_CaloIdL_monitoring.nelectrons = cms.uint32(1)
Mu43NoFiltersNoVtx_Photon43_CaloIdL_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu43NoFiltersNoVtx_Photon43_CaloIdL_v*")
Mu43NoFiltersNoVtx_Photon43_CaloIdL_monitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET120_PFMHT120_IDTight_v*","HLT_PFMETTypeOne120_PFMHT120_IDTight_v*","HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*")



Mu43NoFiltersNoVtx_Photon43_CaloIdL_MuLeg_monitoring = hltMuonmonitoring.clone()
Mu43NoFiltersNoVtx_Photon43_CaloIdL_MuLeg_monitoring.FolderName = cms.string('HLT/EXO/Mu43NoFiltersNoVtx_Photon43_CaloIdL_MuLeg/')
Mu43NoFiltersNoVtx_Photon43_CaloIdL_MuLeg_monitoring.nmuons = cms.uint32(1)
Mu43NoFiltersNoVtx_Photon43_CaloIdL_MuLeg_monitoring.nelectrons = cms.uint32(1)
Mu43NoFiltersNoVtx_Photon43_CaloIdL_MuLeg_monitoring.eleSelection = cms.string('pt > 43')
Mu43NoFiltersNoVtx_Photon43_CaloIdL_MuLeg_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu43NoFiltersNoVtx_Photon43_CaloIdL_v*")
Mu43NoFiltersNoVtx_Photon43_CaloIdL_MuLeg_monitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET120_PFMHT120_IDTight_v*","HLT_PFMETTypeOne120_PFMHT120_IDTight_v*","HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*")




Mu43NoFiltersNoVtx_Photon43_CaloIdL_EleLeg_monitoring = hltMuonmonitoring.clone()
Mu43NoFiltersNoVtx_Photon43_CaloIdL_EleLeg_monitoring.FolderName = cms.string('HLT/EXO/Mu43NoFiltersNoVtx_Photon43_CaloIdL_EleLeg/')
Mu43NoFiltersNoVtx_Photon43_CaloIdL_EleLeg_monitoring.nmuons = cms.uint32(1)
Mu43NoFiltersNoVtx_Photon43_CaloIdL_EleLeg_monitoring.nelectrons = cms.uint32(1)
Mu43NoFiltersNoVtx_Photon43_CaloIdL_EleLeg_monitoring.muonSelection = cms.string('pt > 43')
Mu43NoFiltersNoVtx_Photon43_CaloIdL_EleLeg_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu43NoFiltersNoVtx_Photon43_CaloIdL_v*")
Mu43NoFiltersNoVtx_Photon43_CaloIdL_EleLeg_monitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET120_PFMHT120_IDTight_v*","HLT_PFMETTypeOne120_PFMHT120_IDTight_v*","HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*")


Mu48NoFiltersNoVtx_Photon48_CaloIdL_monitoring = hltMuonmonitoring.clone()
Mu48NoFiltersNoVtx_Photon48_CaloIdL_monitoring.FolderName = cms.string('HLT/EXO/Mu48NoFiltersNoVtx_Photon48_CaloIdL/')
Mu48NoFiltersNoVtx_Photon48_CaloIdL_monitoring.nmuons = cms.uint32(1)
Mu48NoFiltersNoVtx_Photon48_CaloIdL_monitoring.nelectrons = cms.uint32(1)
Mu48NoFiltersNoVtx_Photon48_CaloIdL_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu48NoFiltersNoVtx_Photon48_CaloIdL_v*")
Mu48NoFiltersNoVtx_Photon48_CaloIdL_monitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET120_PFMHT120_IDTight_v*","HLT_PFMETTypeOne120_PFMHT120_IDTight_v*","HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*")



Mu48NoFiltersNoVtx_Photon48_CaloIdL_MuLeg_monitoring = hltMuonmonitoring.clone()
Mu48NoFiltersNoVtx_Photon48_CaloIdL_MuLeg_monitoring.FolderName = cms.string('HLT/EXO/Mu48NoFiltersNoVtx_Photon48_CaloIdL_MuLeg/')
Mu48NoFiltersNoVtx_Photon48_CaloIdL_MuLeg_monitoring.nmuons = cms.uint32(1)
Mu48NoFiltersNoVtx_Photon48_CaloIdL_MuLeg_monitoring.nelectrons = cms.uint32(1)
Mu48NoFiltersNoVtx_Photon48_CaloIdL_MuLeg_monitoring.eleSelection = cms.string('pt > 48')
Mu48NoFiltersNoVtx_Photon48_CaloIdL_MuLeg_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu48NoFiltersNoVtx_Photon48_CaloIdL_v*")
Mu48NoFiltersNoVtx_Photon48_CaloIdL_MuLeg_monitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET120_PFMHT120_IDTight_v*","HLT_PFMETTypeOne120_PFMHT120_IDTight_v*","HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*")




Mu48NoFiltersNoVtx_Photon48_CaloIdL_EleLeg_monitoring = hltMuonmonitoring.clone()
Mu48NoFiltersNoVtx_Photon48_CaloIdL_EleLeg_monitoring.FolderName = cms.string('HLT/EXO/Mu48NoFiltersNoVtx_Photon48_CaloIdL_EleLeg/')
Mu48NoFiltersNoVtx_Photon48_CaloIdL_EleLeg_monitoring.nmuons = cms.uint32(1)
Mu48NoFiltersNoVtx_Photon48_CaloIdL_EleLeg_monitoring.nelectrons = cms.uint32(1)
Mu48NoFiltersNoVtx_Photon48_CaloIdL_EleLeg_monitoring.muonSelection = cms.string('pt > 48')
Mu48NoFiltersNoVtx_Photon48_CaloIdL_EleLeg_monitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Mu48NoFiltersNoVtx_Photon48_CaloIdL_v*")
Mu48NoFiltersNoVtx_Photon48_CaloIdL_EleLeg_monitoring.denGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET120_PFMHT120_IDTight_v*","HLT_PFMETTypeOne120_PFMHT120_IDTight_v*","HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*")



exoHLTMuonmonitoring = cms.Sequence(
    TrkMu12_DoubleTrkMu5NoFiltersNoVtx_monitoring 
    + TrkMu16_DoubleTrkMu6NoFiltersNoVtx_monitoring
    + TrkMu17_DoubleTrkMu8NoFiltersNoVtx_monitoring
    + DoubleMu43NoFiltersNoVtx_monitoring
    + Mu43NoFiltersNoVtx_Photon43_CaloIdL_monitoring
    + DoubleMu48NoFiltersNoVtx_monitoring
    + Mu48NoFiltersNoVtx_Photon48_CaloIdL_monitoring
    + Mu43NoFiltersNoVtx_Photon43_CaloIdL_MuLeg_monitoring
    + Mu48NoFiltersNoVtx_Photon48_CaloIdL_MuLeg_monitoring
    + Mu43NoFiltersNoVtx_Photon43_CaloIdL_EleLeg_monitoring
    + Mu48NoFiltersNoVtx_Photon48_CaloIdL_EleLeg_monitoring
)





