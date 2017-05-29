import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.JetMETMonitor_cfi import hltJetMETmonitoring

AK8PFJet360_TrimMass30_PromptMonitoring = hltJetMETmonitoring.clone()
AK8PFJet360_TrimMass30_PromptMonitoring.FolderName = cms.string('HLT/B2GMonitor/AK8PFJet360_TrimMass30')
AK8PFJet360_TrimMass30_PromptMonitoring.ptcut = cms.double(360)
AK8PFJet360_TrimMass30_PromptMonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFJet360_TrimMass30_v*")
AK8PFJet360_TrimMass30_PromptMonitoring.pfjets      = cms.InputTag("ak8PFJetsCHS")


b2gMonitorHLT = cms.Sequence(
    AK8PFJet360_TrimMass30_PromptMonitoring
)




