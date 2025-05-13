import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

ScoutingElectronTagProbeAnalysis = DQMEDAnalyzer('ScoutingElectronTagProbeAnalyzer',
                                                 OutputInternalPath = cms.string('/HLT/ScoutingOffline/EGamma/TnP/Tag_ScoutingElectron'),
                                                 BaseTriggerSelection = cms.vstring(["DST_PFScouting_ZeroBias_v", "DST_PFScouting_SingleMuon_v", "DST_PFScouting_DoubleMuon_v", "DST_PFScouting_JetHT_v"]),
                                                 triggerSelection = cms.vstring(["DST_PFScouting_DoubleEG_v", "DST_PFScouting_SinglePhotonEB_v"]),
                                                 finalfilterSelection = cms.vstring(["hltDoubleEG11CaloIdLHEFilter", "hltEG30EBTightIDTightIsoTrackIsoFilter"]), # Must align with triggerSelection
                                                 TriggerResultTag   = cms.InputTag("TriggerResults", "", "HLT"),
                                                 TriggerObjects     = cms.InputTag("slimmedPatTrigger"),
                                                 ElectronCollection = cms.InputTag('slimmedElectrons'),
                                                 ScoutingElectronCollection = cms.InputTag('hltScoutingEgammaPacker')
                                                 )

scoutingMonitoringTagProbe = cms.Sequence(ScoutingElectronTagProbeAnalysis) # * ScoutingElectronEfficiencySummary)
