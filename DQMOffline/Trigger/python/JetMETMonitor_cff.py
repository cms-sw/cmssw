import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.JetMETMonitor_cfi import hltJetMETmonitoring

# HLT_PFJet450
PFJet450_Prommonitoring = hltJetMETmonitoring.clone()
PFJet450_Prommonitoring.FolderName = cms.string('HLT/JetMETMonitor/PFJet450/')
PFJet450_Prommonitoring.ptcut = cms.double(450)
PFJet450_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFJet450_v*")

# HLT_PFJet40
PFJet40_Prommonitoring = hltJetMETmonitoring.clone()
PFJet40_Prommonitoring.FolderName = cms.string('HLT/JetMETMonitor/PFJet40/')
PFJet40_Prommonitoring.ptcut = cms.double(40)
PFJet40_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFJet40_v*")
# HLT_PFJet60
PFJet60_Prommonitoring = hltJetMETmonitoring.clone()
PFJet60_Prommonitoring.FolderName = cms.string('HLT/JetMETMonitor/PFJet60/')
PFJet60_Prommonitoring.ptcut = cms.double(60)
PFJet60_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFJet60_v*")
# HLT_PFJet80
PFJet80_Prommonitoring = hltJetMETmonitoring.clone()
PFJet80_Prommonitoring.FolderName = cms.string('HLT/JetMETMonitor/PFJet80/')
PFJet80_Prommonitoring.ptcut = cms.double(80)
PFJet80_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFJet80_v*")

# HLT_AK8PFJet500
AK8PFJet500_Prommonitoring = hltJetMETmonitoring.clone()
AK8PFJet500_Prommonitoring.FolderName = cms.string('HLT/JetMETMonitor/HLT_AK8PFJet500/')
AK8PFJet500_Prommonitoring.hltpfjets = cms.InputTag("hltAK8PFJets")
AK8PFJet500_Prommonitoring.ptcut = cms.double(500)
AK8PFJet500_Prommonitoring.ispfjettrg = cms.bool(True)
AK8PFJet500_Prommonitoring.iscalojettrg = cms.bool(False)
AK8PFJet500_Prommonitoring.ismettrg = cms.bool(False)
AK8PFJet500_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFJet500_v*")

# HLT_CaloJet500_NoJetID
CaloJet500_NoJetID_Prommonitoring = hltJetMETmonitoring.clone()
CaloJet500_NoJetID_Prommonitoring.FolderName = cms.string('HLT/JetMETMonitor/HLT_CaloJet500_NoJetID/')
CaloJet500_NoJetID_Prommonitoring.ptcut = cms.double(500)
CaloJet500_NoJetID_Prommonitoring.ispfjettrg = cms.bool(False)
CaloJet500_NoJetID_Prommonitoring.iscalojettrg = cms.bool(True)
CaloJet500_NoJetID_Prommonitoring.ismettrg = cms.bool(False)
CaloJet500_NoJetID_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_CaloJet500_NoJetID_v*")

# HLT_PFMET170_HBHECleaned
PFMET170_HBHECleaned_Prommonitoring = hltJetMETmonitoring.clone()
PFMET170_HBHECleaned_Prommonitoring.FolderName = cms.string('HLT/JetMETMonitor/HLT_PFMET170_HBHECleaned/')
PFMET170_HBHECleaned_Prommonitoring.ptcut = cms.double(170)
PFMET170_HBHECleaned_Prommonitoring.ispfjettrg = cms.bool(False)
PFMET170_HBHECleaned_Prommonitoring.iscalojettrg = cms.bool(False)
PFMET170_HBHECleaned_Prommonitoring.ismettrg = cms.bool(True)
PFMET170_HBHECleaned_Prommonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFMET170_HBHECleaned_v*")

# HLT_PFMET170_HBHECleaned
JetFrac_Prommonitoring = hltJetMETmonitoring.clone()
JetFrac_Prommonitoring.FolderName = cms.string('HLT/JetMETMonitor/JetEnFraction/')
JetFrac_Prommonitoring.ptcut = cms.double(20)
JetFrac_Prommonitoring.ispfjettrg = cms.bool(False)
JetFrac_Prommonitoring.iscalojettrg = cms.bool(False)
JetFrac_Prommonitoring.ismettrg = cms.bool(False)
JetFrac_Prommonitoring.isjetFrac = cms.bool(True)


HLTJetMETmonitoring = cms.Sequence(
    PFJet450_Prommonitoring
    *PFJet40_Prommonitoring    
    *PFJet60_Prommonitoring    
    *PFJet80_Prommonitoring    
    *AK8PFJet500_Prommonitoring    
    *CaloJet500_NoJetID_Prommonitoring 
    *PFMET170_HBHECleaned_Prommonitoring
    *JetFrac_Prommonitoring
)
