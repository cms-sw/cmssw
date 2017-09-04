import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.JetMonitor_cfi import hltJetMETmonitoring
from DQMOffline.Trigger.HTMonitor_cfi import hltHTmonitoring
from DQMOffline.Trigger.B2GTnPMonitor_cfi import B2GegmGsfElectronIDsForDQM,B2GegHLTDQMOfflineTnPSource
from DQMOffline.Trigger.topDiLeptonHLTEventDQM_cfi import topDiLeptonHLTOfflineDQM


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


AK8PFJet360_TrimMass30_PromptMonitoring = hltJetMETmonitoring.clone()
AK8PFJet360_TrimMass30_PromptMonitoring.FolderName = cms.string('HLT/B2GMonitor/AK8PFJet360_TrimMass30')
AK8PFJet360_TrimMass30_PromptMonitoring.ptcut = cms.double(360)
AK8PFJet360_TrimMass30_PromptMonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFJet360_TrimMass30_v*")

AK8PFJet380_TrimMass30_PromptMonitoring = hltJetMETmonitoring.clone()
AK8PFJet380_TrimMass30_PromptMonitoring.FolderName = cms.string('HLT/B2GMonitor/AK8PFJet380_TrimMass30')
AK8PFJet380_TrimMass30_PromptMonitoring.ptcut = cms.double(380)
AK8PFJet380_TrimMass30_PromptMonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFJet380_TrimMass30_v*")

AK8PFJet400_TrimMass30_PromptMonitoring = hltJetMETmonitoring.clone()
AK8PFJet400_TrimMass30_PromptMonitoring.FolderName = cms.string('HLT/B2GMonitor/AK8PFJet400_TrimMass30')
AK8PFJet400_TrimMass30_PromptMonitoring.ptcut = cms.double(400)
AK8PFJet400_TrimMass30_PromptMonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFJet400_TrimMass30_v*")

AK8PFJet420_TrimMass30_PromptMonitoring = hltJetMETmonitoring.clone()
AK8PFJet420_TrimMass30_PromptMonitoring.FolderName = cms.string('HLT/B2GMonitor/AK8PFJet420_TrimMass30')
AK8PFJet420_TrimMass30_PromptMonitoring.ptcut = cms.double(420)
AK8PFJet420_TrimMass30_PromptMonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFJet420_TrimMass30_v*")


b2gDileptonHLTOfflineDQM = topDiLeptonHLTOfflineDQM.clone()
b2gDileptonHLTOfflineDQM.setup.directory = cms.string('HLT/B2GHLTOffline/Dileptonic/CrossTriggers')
b2gDileptonHLTOfflineDQM.setup.triggerExtras.pathsELECMU = cms.vstring(['HLT_Mu37_Ele27_CaloIdL_MW_v','HLT_Mu27_Ele37_CaloIdL_MW_v'])
b2gDileptonHLTOfflineDQM.setup.triggerExtras.pathsDIMUON = cms.vstring([''])
b2gDileptonHLTOfflineDQM.setup.triggerExtras.pathsDIELEC = cms.vstring([''])
b2gDileptonHLTOfflineDQM.preselection.trigger.select = cms.vstring(['HLT_Mu37_Ele27_CaloIdL_MW_v','HLT_Mu27_Ele37_CaloIdL_MW_v'])

b2gDimuonHLTOfflineDQM = topDiLeptonHLTOfflineDQM.clone()
b2gDimuonHLTOfflineDQM.setup.directory = cms.string('HLT/B2GHLTOffline/Dileptonic/Dimuon')
b2gDimuonHLTOfflineDQM.setup.triggerExtras.pathsELECMU = cms.vstring([''])
b2gDimuonHLTOfflineDQM.setup.triggerExtras.pathsDIMUON = cms.vstring(['HLT_Mu37_TkMu27_v'])
b2gDimuonHLTOfflineDQM.setup.triggerExtras.pathsDIELEC = cms.vstring([''])
b2gDimuonHLTOfflineDQM.preselection.trigger.select = cms.vstring(['HLT_Mu37_TkMu27'])



b2gMonitorHLT = cms.Sequence(
    AK8PFHT750_TrimMass50_HTmonitoring +
    AK8PFHT800_TrimMass50_HTmonitoring + 
    AK8PFHT850_TrimMass50_HTmonitoring + 
    AK8PFHT900_TrimMass50_HTmonitoring +
    AK8PFJet360_TrimMass30_PromptMonitoring + 
    AK8PFJet380_TrimMass30_PromptMonitoring + 
    AK8PFJet400_TrimMass30_PromptMonitoring + 
    AK8PFJet420_TrimMass30_PromptMonitoring +
    B2GegmGsfElectronIDsForDQM*
    B2GegHLTDQMOfflineTnPSource*
    b2gDileptonHLTOfflineDQM*
    b2gDimuonHLTOfflineDQM
)
