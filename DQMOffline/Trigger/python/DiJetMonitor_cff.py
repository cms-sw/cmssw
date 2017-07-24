import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.DiJetMonitor_cfi import DiPFjetAve40_Prommonitoring
### HLT_DiJet Triggers ###
# DiPFjetAve60
DiPFjetAve60_Prommonitoring = DiPFjetAve40_Prommonitoring.clone()
DiPFjetAve60_Prommonitoring.FolderName = cms.string('HLT/JetMET/HLT_DiPFJetAve60/')
DiPFjetAve60_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DiPFJetAve60_v*")

# DiPFjetAve80
DiPFjetAve80_Prommonitoring = DiPFjetAve40_Prommonitoring.clone()
DiPFjetAve80_Prommonitoring.FolderName = cms.string('HLT/JetMET/HLT_DiPFJetAve80/')
DiPFjetAve80_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DiPFJetAve80_v*")

# DiPFjetAve140
DiPFjetAve140_Prommonitoring = DiPFjetAve40_Prommonitoring.clone()
DiPFjetAve140_Prommonitoring.FolderName = cms.string('HLT/JetMET/HLT_DiPFJetAve140/')
DiPFjetAve140_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DiPFJetAve140_v*")

# DiPFjetAve200
DiPFjetAve200_Prommonitoring = DiPFjetAve40_Prommonitoring.clone()
DiPFjetAve200_Prommonitoring.FolderName = cms.string('HLT/JetMET/HLT_DiPFJetAve200/')
DiPFjetAve200_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DiPFJetAve200_v*")

# DiPFjetAve260
DiPFjetAve260_Prommonitoring = DiPFjetAve40_Prommonitoring.clone()
DiPFjetAve260_Prommonitoring.FolderName = cms.string('HLT/JetMET/HLT_DiPFJetAve260/')
DiPFjetAve260_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DiPFJetAve260_v*")

# DiPFjetAve320
DiPFjetAve320_Prommonitoring = DiPFjetAve40_Prommonitoring.clone()
DiPFjetAve320_Prommonitoring.FolderName = cms.string('HLT/JetMET/HLT_DiPFJetAve320/')
DiPFjetAve320_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DiPFJetAve320_v*")

# DiPFjetAve400
DiPFjetAve400_Prommonitoring = DiPFjetAve40_Prommonitoring.clone()
DiPFjetAve400_Prommonitoring.FolderName = cms.string('HLT/JetMET/HLT_DiPFJetAve400/')
DiPFjetAve400_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DiPFJetAve400_v*")

# DiPFjetAve500
DiPFjetAve500_Prommonitoring = DiPFjetAve40_Prommonitoring.clone()
DiPFjetAve500_Prommonitoring.FolderName = cms.string('HLT/JetMET/HLT_DiPFJetAve500/')
DiPFjetAve500_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DiPFJetAve500_v*")

# HLT_DiPFJetAve60_HFJEC
DiPFJetAve60_HFJEC_Prommonitoring = DiPFjetAve40_Prommonitoring.clone()
DiPFJetAve60_HFJEC_Prommonitoring.FolderName = cms.string('HLT/JetMET/HLT_DiPFJetAve60_HFJEC/')
DiPFJetAve60_HFJEC_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DiPFJetAve60_HFJEC_v*")

# HLT_DiPFJetAve80_HFJEC
DiPFJetAve80_HFJEC_Prommonitoring = DiPFjetAve40_Prommonitoring.clone()
DiPFJetAve80_HFJEC_Prommonitoring.FolderName = cms.string('HLT/JetMET/HLT_DiPFJetAve80_HFJEC/')
DiPFJetAve80_HFJEC_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DiPFJetAve80_HFJEC_v*")

# HLT_DiPFJetAve100_HFJEC
DiPFJetAve100_HFJEC_Prommonitoring = DiPFjetAve40_Prommonitoring.clone()
DiPFJetAve100_HFJEC_Prommonitoring.FolderName = cms.string('HLT/JetMET/HLT_DiPFJetAve100_HFJEC/')
DiPFJetAve100_HFJEC_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DiPFJetAve100_HFJEC_v*")

# HLT_DiPFJetAve160_HFJEC
DiPFJetAve160_HFJEC_Prommonitoring = DiPFjetAve40_Prommonitoring.clone()
DiPFJetAve160_HFJEC_Prommonitoring.FolderName = cms.string('HLT/JetMET/HLT_DiPFJetAve160_HFJEC/')
DiPFJetAve160_HFJEC_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DiPFJetAve160_HFJEC_v*")

# HLT_DiPFJetAve220_HFJEC
DiPFJetAve220_HFJEC_Prommonitoring = DiPFjetAve40_Prommonitoring.clone()
DiPFJetAve220_HFJEC_Prommonitoring.FolderName = cms.string('HLT/JetMET/HLT_DiPFJetAve220_HFJEC/')
DiPFJetAve220_HFJEC_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DiPFJetAve220_HFJEC_v*")

# HLT_DiPFJetAve300_HFJEC
DiPFJetAve300_HFJEC_Prommonitoring = DiPFjetAve40_Prommonitoring.clone()
DiPFJetAve300_HFJEC_Prommonitoring.FolderName = cms.string('HLT/JetMET/HLT_DiPFJetAve300_HFJEC/')
DiPFJetAve300_HFJEC_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_DiPFJetAve300_HFJEC_v*")

HLTDiJetmonitoring = cms.Sequence(
    DiPFjetAve40_Prommonitoring
    *DiPFjetAve60_Prommonitoring
    *DiPFjetAve80_Prommonitoring
    *DiPFjetAve140_Prommonitoring
    *DiPFjetAve200_Prommonitoring
    *DiPFjetAve260_Prommonitoring
    *DiPFjetAve320_Prommonitoring
    *DiPFjetAve400_Prommonitoring
    *DiPFjetAve500_Prommonitoring
    *DiPFJetAve60_HFJEC_Prommonitoring
    *DiPFJetAve80_HFJEC_Prommonitoring
    *DiPFJetAve100_HFJEC_Prommonitoring
    *DiPFJetAve160_HFJEC_Prommonitoring
    *DiPFJetAve220_HFJEC_Prommonitoring
    *DiPFJetAve300_HFJEC_Prommonitoring
)

