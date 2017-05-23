import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.HTMonitor_cfi import hltHTmonitoring

# HLT_PFMETNoMu90_PFMHTNoMu90_IDTight
PFMETNoMu90_PFMHTNoMu90_HTmonitoring = hltHTmonitoring.clone()
PFMETNoMu90_PFMHTNoMu90_HTmonitoring.FolderName = cms.string('HLT/HT/PFMETNoMu90/')
PFMETNoMu90_PFMHTNoMu90_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMETNoMu90_PFMHTNoMu90_IDTight_v*")

# HLT_PFMETNoMu120_PFMHTNoMu120_IDTight
PFMETNoMu120_PFMHTNoMu120_HTmonitoring = hltHTmonitoring.clone()
PFMETNoMu120_PFMHTNoMu120_HTmonitoring.FolderName = cms.string('HLT/HT/PFMETNoMu120/')
PFMETNoMu120_PFMHTNoMu120_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v*")

# HLT_MET200
MET200_HTmonitoring = hltHTmonitoring.clone()
MET200_HTmonitoring.FolderName = cms.string('HLT/HT/MET200/')
MET200_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_MET200_v*")

# HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight
MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_HTmonitoring = hltHTmonitoring.clone()
MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_HTmonitoring.FolderName = cms.string('HLT/HT/MonoCentralPFJet80_PFMETNoMu120/')
MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*")

# HLT_MonoCentralPFJet80_PFMETNoMu90_PFMHTNoMu90_IDTight
MonoCentralPFJet80_PFMETNoMu90_PFMHTNoMu90_HTmonitoring = hltHTmonitoring.clone()
MonoCentralPFJet80_PFMETNoMu90_PFMHTNoMu90_HTmonitoring.FolderName = cms.string('HLT/HT/MonoCentralPFJet80_PFMETNoMu90/')
MonoCentralPFJet80_PFMETNoMu90_PFMHTNoMu90_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_MonoCentralPFJet80_PFMETNoMu110_PFMHTNoMu110_IDTight_v*")

exoHLTHTmonitoring = cms.Sequence(
    PFMETNoMu90_PFMHTNoMu90_HTmonitoring
    + PFMETNoMu120_PFMHTNoMu120_HTmonitoring
    + MET200_HTmonitoring
    + MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_HTmonitoring
    + MonoCentralPFJet80_PFMETNoMu90_PFMHTNoMu90_HTmonitoring
)

