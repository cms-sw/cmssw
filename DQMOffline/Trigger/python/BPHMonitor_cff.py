import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.BPHMonitor_cfi import hltBPHmonitoring

# HLT_PFMETNoMu90_PFMHTNoMu90_IDTight
Dimuon20_Jpsi_BPHMonitoring = hltBPHmonitoring.clone()
Dimuon20_Jpsi_BPHMonitoring.FolderName = cms.string('HLT/BPH/DiMu20_Jpsi/')
Dimuon20_Jpsi_BPHMonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon20_Jpsi")

Dimuon10_Jpsi_Barrel_BPHMonitoring = hltBPHmonitoring.clone()
Dimuon10_Jpsi_Barrel_BPHMonitoring.FolderName = cms.string('HLT/BPH/DiMu10_Jpsi_Barrel/')
Dimuon10_Jpsi_Barrel_BPHMonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon10_Jpsi_Barrel")


# HLT_PFMETNoMu120_PFMHTNoMu120_IDTight
DoubleMu4_3_Jpsi_Displaced_BPHMonitoring = hltBPHmonitoring.clone()
DoubleMu4_3_Jpsi_Displaced_BPHMonitoring.FolderName = cms.string('HLT/BPH/DoubleMu4_3_Jpsi_Displaced/')
DoubleMu4_3_Jpsi_Displaced_BPHMonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleMu4_3_Jpsi_Displaced")

DoubleMu4_JpsiTrk_Displaced_BPHMonitoring = hltBPHmonitoring.clone()
DoubleMu4_JpsiTrk_Displaced_BPHMonitoring.FolderName = cms.string('HLT/BPH/DoubleMu4_JpsiTrk_Displaced/')
DoubleMu4_JpsiTrk_Displaced_BPHMonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleMu4_JpsiTrk_Displaced")

bphHLTmonitoring = cms.Sequence(
    Dimuon20_Jpsi_BPHMonitoring
    + Dimuon10_Jpsi_Barrel_BPHMonitoring
    + DoubleMu4_3_Jpsi_Displaced_BPHMonitoring
    + DoubleMu4_JpsiTrk_Displaced_BPHMonitoring
)


bphMonitorHLT = cms.Sequence(
    bphHLTmonitoring
)

