import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.BPHMonitor_cfi import hltBPHmonitoring
# Tau3Mu
from DQMOffline.Trigger.Tau3MuMonitor_cff import *

# HLT_PFMETNoMu90_PFMHTNoMu90_IDTight
Dimuon20_Jpsi_BPHMonitoring = hltBPHmonitoring.clone()
Dimuon20_Jpsi_BPHMonitoring.FolderName = cms.string('HLT/BPH/DiMu20_Jpsi/')
Dimuon20_Jpsi_BPHMonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon20_Jpsi")
Dimuon20_Jpsi_BPHMonitoring.muoSelection = cms.string("abs(eta)<2.4 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
#muoSelection
Dimuon10_Jpsi_Barrel_BPHMonitoring = hltBPHmonitoring.clone()
Dimuon10_Jpsi_Barrel_BPHMonitoring.FolderName = cms.string('HLT/BPH/DiMu10_Jpsi_Barrel/')
Dimuon10_Jpsi_Barrel_BPHMonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_Dimuon10_Jpsi_Barrel")
Dimuon10_Jpsi_Barrel_BPHMonitoring.muoSelection = cms.string("abs(eta)<1.4 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")

# HLT_PFMETNoMu120_PFMHTNoMu120_IDTight
DoubleMu4_3_Jpsi_Displaced_BPHMonitoring = hltBPHmonitoring.clone()
DoubleMu4_3_Jpsi_Displaced_BPHMonitoring.FolderName = cms.string('HLT/BPH/DoubleMu4_3_Jpsi_Displaced/')
DoubleMu4_3_Jpsi_Displaced_BPHMonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleMu4_3_Jpsi_Displaced")
DoubleMu4_3_Jpsi_Displaced_BPHMonitoring.muoSelection = cms.string("pt>3 & abs(eta)<2.4 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")

DoubleMu4_JpsiTrk_Displaced_BPHMonitoring = hltBPHmonitoring.clone()
DoubleMu4_JpsiTrk_Displaced_BPHMonitoring.FolderName = cms.string('HLT/BPH/DoubleMu4_JpsiTrk_Displaced/')
DoubleMu4_JpsiTrk_Displaced_BPHMonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DoubleMu4_JpsiTrk_Displaced")
DoubleMu4_JpsiTrk_Displaced_BPHMonitoring.muoSelection = cms.string("pt>4 & abs(eta)<2.4 & isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & innerTrack.hitPattern.numberOfValidPixelHits>0 ")
bphHLTmonitoring = cms.Sequence(
    Dimuon20_Jpsi_BPHMonitoring
    + Dimuon10_Jpsi_Barrel_BPHMonitoring
    + DoubleMu4_3_Jpsi_Displaced_BPHMonitoring
    + DoubleMu4_JpsiTrk_Displaced_BPHMonitoring
)


bphMonitorHLT = cms.Sequence(
    bphHLTmonitoring * 
    tau3MuMonitorHLT    
)

