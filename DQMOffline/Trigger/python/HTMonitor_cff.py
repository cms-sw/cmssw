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

# HLT_PFHT350MinPFJet15
PFHT350MinPFJet15_HTmonitoring = hltHTmonitoring.clone()
PFHT350MinPFJet15_HTmonitoring.FolderName = cms.string('HLT/HT/HLT_PFHT350MinPFJet15/')
PFHT350MinPFJet15_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT350MinPFJet15_v*")
PFHT350MinPFJet15_HTmonitoring.jetSelection      = cms.string("pt > 15")

# HLT_PFHT500_PFMET100_PFMHT100_IDTight                                                                                                                                                          
PFHT500_PFMET100_PFMHT100_HTmonitoring = hltHTmonitoring.clone()
PFHT500_PFMET100_PFMHT100_HTmonitoring.FolderName = cms.string('HLT/MET/PFHT500_PFMET100_PFMHT100/')
PFHT500_PFMET100_PFMHT100_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT500_PFMET100_PFMHT100_IDTight_v*")
PFHT500_PFMET100_PFMHT100_HTmonitoring.metSelection      = cms.string("pt > 200")
PFHT500_PFMET100_PFMHT100_HTmonitoring.jetSelection      = cms.string("pt > 0")

# HLT_PFHT500_PFMET110_PFMHT110_IDTight                                                                                                                                                         
PFHT500_PFMET110_PFMHT110_HTmonitoring = hltHTmonitoring.clone()
PFHT500_PFMET110_PFMHT110_HTmonitoring.FolderName = cms.string('HLT/MET/PFHT500_PFMET110_PFMHT110/')
PFHT500_PFMET110_PFMHT110_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT500_PFMET110_PFMHT110_IDTight_v*")
PFHT500_PFMET110_PFMHT110_HTmonitoring.metSelection      = cms.string("pt > 210")
PFHT500_PFMET110_PFMHT110_HTmonitoring.jetSelection      = cms.string("pt > 0")

# HLT_PFHT700_PFMET85_PFMHT85_IDTight                                                                                                                                                            
PFHT700_PFMET85_PFMHT85_HTmonitoring = hltHTmonitoring.clone()
PFHT700_PFMET85_PFMHT85_HTmonitoring.FolderName = cms.string('HLT/MET/PFHT700_PFMET85_PFMHT85/')
PFHT700_PFMET85_PFMHT85_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT700_PFMET85_PFMHT85_IDTight_v*")
PFHT700_PFMET85_PFMHT85_HTmonitoring.metSelection      = cms.string("pt > 185")
PFHT700_PFMET85_PFMHT85_HTmonitoring.jetSelection      = cms.string("pt > 0")

# HLT_PFHT700_PFMET95_PFMHT95_IDTight                                                                                                                                                            
PFHT700_PFMET95_PFMHT95_HTmonitoring = hltHTmonitoring.clone()
PFHT700_PFMET95_PFMHT95_HTmonitoring.FolderName = cms.string('HLT/MET/PFHT700_PFMET95_PFMHT95/')
PFHT700_PFMET95_PFMHT95_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT700_PFMET95_PFMHT95_IDTight_v*")
PFHT700_PFMET95_PFMHT95_HTmonitoring.metSelection      = cms.string("pt > 195")
PFHT700_PFMET95_PFMHT95_HTmonitoring.jetSelection      = cms.string("pt > 0")

# HLT_PFHT800_PFMET75_PFMHT75_IDTight                                                                                                                                                            
PFHT800_PFMET75_PFMHT75_HTmonitoring = hltHTmonitoring.clone()
PFHT800_PFMET75_PFMHT75_HTmonitoring.FolderName = cms.string('HLT/MET/PFHT800_PFMET75_PFMHT75/')
PFHT800_PFMET75_PFMHT75_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT800_PFMET75_PFMHT75_IDTight_v*")
PFHT800_PFMET75_PFMHT75_HTmonitoring.metSelection      = cms.string("pt > 175")
PFHT800_PFMET75_PFMHT75_HTmonitoring.jetSelection      = cms.string("pt > 0")

# HLT_PFHT800_PFMET85_PFMHT85_IDTight                                                                                                                                                          
PFHT800_PFMET85_PFMHT85_HTmonitoring = hltHTmonitoring.clone()
PFHT800_PFMET85_PFMHT85_HTmonitoring.FolderName = cms.string('HLT/MET/PFHT800_PFMET85_PFMHT85/')
PFHT800_PFMET85_PFMHT85_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT800_PFMET85_PFMHT85_IDTight_v*")
PFHT800_PFMET85_PFMHT85_HTmonitoring.metSelection      = cms.string("pt > 185")
PFHT800_PFMET85_PFMHT85_HTmonitoring.jetSelection      = cms.string("pt > 0")

# HLT_PFHT1050_v7
PFHT1050_HTmonitoring = hltHTmonitoring.clone()
PFHT1050_HTmonitoring.FolderName = cms.string('HLT/MET/PFHT1050/')
PFHT1050_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT1050_v*")
PFHT1050_HTmonitoring.jetSelection      = cms.string("pt > 0")


exoHLTHTmonitoring = cms.Sequence(
    PFMETNoMu90_PFMHTNoMu90_HTmonitoring
    + PFMETNoMu120_PFMHTNoMu120_HTmonitoring
    + MET200_HTmonitoring
    + MonoCentralPFJet80_PFMETNoMu120_PFMHTNoMu120_HTmonitoring
    + MonoCentralPFJet80_PFMETNoMu90_PFMHTNoMu90_HTmonitoring
    + PFHT350MinPFJet15_HTmonitoring
    + PFHT500_PFMET100_PFMHT100_HTmonitoring
    + PFHT500_PFMET110_PFMHT110_HTmonitoring
    + PFHT700_PFMET85_PFMHT85_HTmonitoring
    + PFHT700_PFMET95_PFMHT95_HTmonitoring
    + PFHT800_PFMET75_PFMHT75_HTmonitoring
    + PFHT800_PFMET85_PFMHT85_HTmonitoring
    + PFHT1050_HTmonitoring
)

