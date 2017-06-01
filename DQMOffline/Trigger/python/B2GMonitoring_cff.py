import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.JetMETMonitor_cfi import hltJetMETmonitoring
from DQMOffline.Trigger.HTMonitor_cfi import hltHTmonitoring

# HLT_AK8PFHT700_TrimR0p1PT0p03Mass50                                                                                                                                         
AK8PFHT700_HTmonitoring = hltHTmonitoring.clone()
AK8PFHT700_HTmonitoring.FolderName = cms.string('HLT/B2GMonitor/AK8PFHT700_TrimR0p1PT0p03Mass50')
AK8PFHT700_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFHT700_TrimR0p1PT0p03Mass50_v*")
AK8PFHT700_HTmonitoring.jetSelection      = cms.string("pt > 0")
AK8PFHT700_HTmonitoring.jetSelection_HT = cms.string("pt > 10 && eta < 2.5")

# HLT_AK8PFJet360_TrimMass30                                                                                                                                                  
AK8PFJet360_TrimMass30_PromptMonitoring = hltJetMETmonitoring.clone()
AK8PFJet360_TrimMass30_PromptMonitoring.FolderName = cms.string('HLT/B2GMonitor/AK8PFJet360_TrimMass30')
AK8PFJet360_TrimMass30_PromptMonitoring.ptcut = cms.double(360)
AK8PFJet360_TrimMass30_Promptonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFJet360_TrimMass30_v*")
AK8PFJet360_TrimMass30_PromptMonitoring.pfjets      = cms.InputTag("ak8PFJetsCHS")
AK8PFJet360_TrimMass30_PromptMonitoring.jetSelection      = cms.string("pt > 0")
AK8PFJet360_TrimMass30_PromptMonitoring.jetSelection_HT= cms.string("pt > 10 && eta < 2.5")


# HLT_PFHT650_WideJetMJJ900DEtaJJ1p5                                                                                                                   
PFHT650_WideJetMJJ900DEtaJJ1p5_HTmonitoring = hltHTmonitoring.clone()                        
PFHT650_WideJetMJJ900DEtaJJ1p5_HTmonitoring.FolderName = cms.string('HLT/B2GMonitor/PFHT650_WideJetMJJ900DEtaJJ1p5')
PFHT650_WideJetMJJ900DEtaJJ1p5_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT650_WideJetMJJ900DEtaJJ1p5_v*")
PFHT650_WideJetMJJ900DEtaJJ1p5_HTmonitoring.jetSelection = cms.string("pt > 0")
PFHT650_WideJetMJJ900DEtaJJ1p5_HTmonitoring.jetSelection_HT =  cms.string("pt > 10 && eta < 2.5")

# HLT_PFHT800                                                                                                                                          
PFHT800_HTmonitoring = hltHTmonitoring.clone()
PFHT800_HTmonitoring.FolderName = cms.string('HLT/B2GMonitor/PFHT800')
PFHT800_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT800_v*")
PFHT800_HTmonitoring.jetSelection = cms.string("pt > 0") 
PFHT800_HTmonitoring.jetSelection_HT =  cms.string("pt > 10 && eta < 2.5")

b2gMonitorHLT = cms.Sequence(
    AK8PFHT700_HTmonitoring
    AK8PFJet360_TrimMass30_PromptMonitoring
    PFHT650_WideJetMJJ900DEtaJJ1p5_HTmonitoring
    PFHT800_HTmonitoring
)




