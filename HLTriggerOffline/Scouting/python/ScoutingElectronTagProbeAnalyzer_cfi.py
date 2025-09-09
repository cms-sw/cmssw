import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

ScoutingElectronTagProbeAnalysisOffline = DQMEDAnalyzer('ScoutingElectronTagProbeAnalyzer',
                                                 OutputInternalPath = cms.string('/HLT/ScoutingOffline/EGamma/TnP/Tag_ScoutingElectron'),
                                                 BaseTriggerSelection = cms.vstring(["DST_PFScouting_ZeroBias_v", "DST_PFScouting_SingleMuon_v", "DST_PFScouting_DoubleMuon_v", "DST_PFScouting_JetHT_v"]),
                                                 triggerSelection = cms.vstring(["DST_PFScouting_DoubleEG_v", "DST_PFScouting_SinglePhotonEB_v"]),
                                                 finalfilterSelection = cms.vstring(["hltDoubleEG11CaloIdLHEFilter", "hltEG30EBTightIDTightIsoTrackIsoFilter"]), # Must align with triggerSelection
                                                 TriggerResultTag   = cms.InputTag("TriggerResults", "", "HLT"),
                                                 TriggerObjects     = cms.InputTag("slimmedPatTrigger"),
                                                 ScoutingElectronCollection = cms.InputTag('hltScoutingEgammaPacker'),
                                                 useOfflineObject = cms.bool(True)
                                                 )

ScoutingElectronTagProbeAnalysisOnline = ScoutingElectronTagProbeAnalysisOffline.clone(
                                                 OutputInternalPath = '/HLT/ScoutingOnline/EGamma/TnP/Tag_ScoutingElectron',
                                                 useOfflineObject = False
                                                 )

scoutingMonitoringTagProbeOffline = cms.Sequence(ScoutingElectronTagProbeAnalysisOffline)
scoutingMonitoringTagProbeOnline = cms.Sequence(ScoutingElectronTagProbeAnalysisOnline)
