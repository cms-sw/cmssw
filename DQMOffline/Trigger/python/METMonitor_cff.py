import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.METMonitor_cfi import hltMETmonitoring

# HLT_PFMETNoMu90_PFMHTNoMu90_IDTight
PFMETNoMu90_PFMHTNoMu90_METmonitoring = hltMETmonitoring.clone()
PFMETNoMu90_PFMHTNoMu90_METmonitoring.FolderName = cms.string('HLT/MET/PFMETNoMu90/')
PFMETNoMu90_PFMHTNoMu90_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMETNoMu90_PFMHTNoMu90_IDTight_v*")

# HLT_PFMETNoMu120_PFMHTNoMu120_IDTight
PFMETNoMu120_PFMHTNoMu120_METmonitoring = hltMETmonitoring.clone()
PFMETNoMu120_PFMHTNoMu120_METmonitoring.FolderName = cms.string('HLT/MET/PFMETNoMu120/')
PFMETNoMu120_PFMHTNoMu120_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v*")

# HLT_MET200
MET200_METmonitoring = hltMETmonitoring.clone()
MET200_METmonitoring.FolderName = cms.string('HLT/MET/MET200/')
MET200_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_MET200_v*")

# HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight
MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_METmonitoring = hltMETmonitoring.clone()
MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_METmonitoring.FolderName = cms.string('HLT/MET/MonoCentralPFJet80_PFMETNoMu120/')
MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_IDTight_v*")

# HLT_MonoCentralPFJet80_PFMETNoMu90_PFMHTNoMu90_IDTight
MonoCentralPFJet80_PFMETNoMu90_PFMHTNoMu90_METmonitoring = hltMETmonitoring.clone()
MonoCentralPFJet80_PFMETNoMu90_PFMHTNoMu90_METmonitoring.FolderName = cms.string('HLT/MET/MonoCentralPFJet80_PFMETNoMu90/')
MonoCentralPFJet80_PFMETNoMu90_PFMHTNoMu90_METmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_MonoCentralPFJet80_PFMETNoMu110_PFMHTNoMu110_IDTight_v*")

exoHLTMETmonitoring = cms.Sequence(
    PFMETNoMu90_PFMHTNoMu90_METmonitoring
    + PFMETNoMu120_PFMHTNoMu120_METmonitoring
    + MET200_METmonitoring
    + MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_METmonitoring
    + MonoCentralPFJet80_PFMETNoMu90_PFMHTNoMu90_METmonitoring
)

