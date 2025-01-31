import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

PatElectronTagProbeAnalysis = DQMEDAnalyzer('PatElectronTagProbeAnalyzer',

    OutputInternalPath = cms.string('/HLT/ScoutingOffline/EGamma/TnP/Tag_PatElectron'),
    TriggerResultTag   = cms.InputTag("TriggerResults", "", "HLT"),
    TriggerObjects     = cms.InputTag("slimmedPatTrigger"),
    ElectronCollection = cms.InputTag('slimmedElectrons'),
    ScoutingElectronCollection = cms.InputTag('hltScoutingEgammaPacker'),
    eleIdMapTight = cms.InputTag('egmGsfElectronIDsForScoutingDQM:cutBasedElectronID-RunIIIWinter22-V1-tight')

)

scoutingMonitoringPatElectronTagProbe = cms.Sequence(PatElectronTagProbeAnalysis)
