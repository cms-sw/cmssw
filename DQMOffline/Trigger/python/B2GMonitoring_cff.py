import FWCore.ParameterSet.Config as cms

# THIS WILL BE UPDATED WHEN 19178 IS MERGED:
#from DQMOffline.Trigger.JetMETMonitor_cfi import hltJetMETmonitoring
from DQMOffline.Trigger.HTMonitor_cfi import hltHTmonitoring
from DQMOffline.Trigger.B2GTnPMonitor_cfi import B2GegmGsfElectronIDsForDQM,B2GegHLTDQMOfflineTnPSource

# B2G triggers:
#HLT_AK8PFHT750_TrimMass50_v*
#HLT_AK8PFJet380_TrimMass30_v*
#HLT_AK8PFHT800_TrimMass50_v*
#HLT_AK8PFJet400_TrimMass30_v*
#HLT_AK8PFHT850_TrimMass50_v*
#HLT_AK8PFJet420_TrimMass30_v*
#HLT_AK8PFHT900_TrimMass50_v*
# HLT_AK8PFHT700_TrimR0p1PT0p03Mass50                                                                                                                                         
AK8PFHT750_TrimMass50_HTmonitoring = hltHTmonitoring.clone()
AK8PFHT750_TrimMass50_HTmonitoring.FolderName = cms.string('HLT/B2GMonitor/AK8PFHT750_TrimMass50')
AK8PFHT750_TrimMass50_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFHT750_TrimMass50_v*")
AK8PFHT750_TrimMass50_HTmonitoring.jets = cms.InputTag("ak8PFJetsCHS")
AK8PFHT750_TrimMass50_HTmonitoring.jetSelection      = cms.string("pt > 0 && eta < 2.5")
AK8PFHT750_TrimMass50_HTmonitoring.jetSelection_HT = cms.string("pt > 200 && eta < 2.5")

AK8PFHT800_TrimMass50_HTmonitoring = hltHTmonitoring.clone()
AK8PFHT800_TrimMass50_HTmonitoring.FolderName = cms.string('HLT/B2GMonitor/AK8PFHT800_TrimMass50')
AK8PFHT800_TrimMass50_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFHT800_TrimMass50_v*")
AK8PFHT800_TrimMass50_HTmonitoring.jets = cms.InputTag("ak8PFJetsCHS")
AK8PFHT800_TrimMass50_HTmonitoring.jetSelection      = cms.string("pt > 0 && eta < 2.5")
AK8PFHT800_TrimMass50_HTmonitoring.jetSelection_HT = cms.string("pt > 200 && eta < 2.5")

AK8PFHT850_TrimMass50_HTmonitoring = hltHTmonitoring.clone()
AK8PFHT850_TrimMass50_HTmonitoring.FolderName = cms.string('HLT/B2GMonitor/AK8PFHT850_TrimMass50')
AK8PFHT850_TrimMass50_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFHT850_TrimMass50_v*")
AK8PFHT850_TrimMass50_HTmonitoring.jets = cms.InputTag("ak8PFJetsCHS")
AK8PFHT850_TrimMass50_HTmonitoring.jetSelection      = cms.string("pt > 0 && eta < 2.5")
AK8PFHT850_TrimMass50_HTmonitoring.jetSelection_HT = cms.string("pt > 200 && eta < 2.5")

AK8PFHT900_TrimMass50_HTmonitoring = hltHTmonitoring.clone()
AK8PFHT900_TrimMass50_HTmonitoring.FolderName = cms.string('HLT/B2GMonitor/AK8PFHT900_TrimMass50')
AK8PFHT900_TrimMass50_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFHT900_TrimMass50_v*")
AK8PFHT900_TrimMass50_HTmonitoring.jets = cms.InputTag("ak8PFJetsCHS")
AK8PFHT900_TrimMass50_HTmonitoring.jetSelection      = cms.string("pt > 0 && eta < 2.5")
AK8PFHT900_TrimMass50_HTmonitoring.jetSelection_HT = cms.string("pt > 200 && eta < 2.5")


# THE FOLLOWING TO BE ADDED WHEN PR 19178 is merged:
'''               
AK8PFJet380_TrimMass30_PromptMonitoring = hltJetMETmonitoring.clone()
AK8PFJet380_TrimMass30_PromptMonitoring.FolderName = cms.string('HLT/B2GMonitor/AK8PFJet380_TrimMass30')
AK8PFJet380_TrimMass30_PromptMonitoring.ptcut = cms.double(380)
AK8PFJet380_TrimMass30_Promptonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFJet380_TrimMass30_v*")
AK8PFJet380_TrimMass30_PromptMonitoring.pfjets      = cms.InputTag("ak8PFJetsCHS")
AK8PFJet380_TrimMass30_PromptMonitoring.jetSelection      = cms.string("pt > 0 && eta < 2.5")
AK8PFJet380_TrimMass30_PromptMonitoring.jetSelection_HT= cms.string("pt > 200 && eta < 2.5")

AK8PFJet400_TrimMass30_PromptMonitoring = hltJetMETmonitoring.clone()
AK8PFJet400_TrimMass30_PromptMonitoring.FolderName = cms.string('HLT/B2GMonitor/AK8PFJet400_TrimMass30')
AK8PFJet400_TrimMass30_PromptMonitoring.ptcut = cms.double(400)
AK8PFJet400_TrimMass30_Promptonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFJet400_TrimMass30_v*")
AK8PFJet400_TrimMass30_PromptMonitoring.pfjets      = cms.InputTag("ak8PFJetsCHS")
AK8PFJet400_TrimMass30_PromptMonitoring.jetSelection      = cms.string("pt > 0 && eta < 2.5")
AK8PFJet400_TrimMass30_PromptMonitoring.jetSelection_HT= cms.string("pt > 200 && eta < 2.5")

AK8PFJet420_TrimMass30_PromptMonitoring = hltJetMETmonitoring.clone()
AK8PFJet420_TrimMass30_PromptMonitoring.FolderName = cms.string('HLT/B2GMonitor/AK8PFJet420_TrimMass30')
AK8PFJet420_TrimMass30_PromptMonitoring.ptcut = cms.double(420)
AK8PFJet420_TrimMass30_Promptonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFJet420_TrimMass30_v*")
AK8PFJet420_TrimMass30_PromptMonitoring.pfjets      = cms.InputTag("ak8PFJetsCHS")
AK8PFJet420_TrimMass30_PromptMonitoring.jetSelection      = cms.string("pt > 0 && eta < 2.5")
AK8PFJet420_TrimMass30_PromptMonitoring.jetSelection_HT= cms.string("pt > 200 && eta < 2.5")
'''


b2gMonitorHLT = cms.Sequence(
    AK8PFHT750_TrimMass50_HTmonitoring +
    AK8PFHT800_TrimMass50_HTmonitoring + 
    AK8PFHT850_TrimMass50_HTmonitoring + 
    AK8PFHT900_TrimMass50_HTmonitoring# +
    #AK8PFJet380_TrimMass30 + 
    #AK8PFJet400_TrimMass30 + 
    #AK8PFJet420_TrimMass30
)


b2gMonitorHLT_elplusJet = cms.Sequence(
    B2GegmGsfElectronIDsForDQM*
    B2GegHLTDQMOfflineTnPSource
)
