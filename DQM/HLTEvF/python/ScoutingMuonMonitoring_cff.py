import FWCore.ParameterSet.Config as cms

## See DQMOffline/HLTScouting/python/HLTScoutingDqmOffline_cff.py

from HLTriggerOffline.Scouting.ScoutingMuonTriggerAnalyzer_cfi import ScoutingMuonTriggerAnalysis_DoubleMu, ScoutingMuonTriggerAnalysis_SingleMu
from HLTriggerOffline.Scouting.ScoutingMuonTagProbeAnalyzer_cfi import ScoutingMuonTagProbeAnalysisNoVtx, ScoutingMuonTagProbeAnalysisVtx
from HLTriggerOffline.Scouting.ScoutingMuonPropertiesMonitoring_cfi import ScoutingMuonPropertiesMonitor

ScoutingMuonTagProbeAnalysisNoVtxOnline = ScoutingMuonTagProbeAnalysisNoVtx.clone(OutputInternalPath = "/HLT/ScoutingOnline/Muons/NoVtx")
ScoutingMuonTagProbeAnalysisVtxOnline = ScoutingMuonTagProbeAnalysisVtx.clone(OutputInternalPath = "/HLT/ScoutingOnline/Muons/Vtx")
ScoutingMuonTriggerAnalysis_DoubleMu = ScoutingMuonTriggerAnalysis_DoubleMu.clone(OutputInternalPath = "/HLT/ScoutingOnline/Muons/L1Efficiency/DoubleMu")
ScoutingMuonTriggerAnalysis_SingleMu = ScoutingMuonTriggerAnalysis_SingleMu.clone(OutputInternalPath = "/HLT/ScoutingOnline/Muons/L1Efficiency/SingleMu")
ScoutingMuonPropertiesMonitorOnline = ScoutingMuonPropertiesMonitor.clone(OutputInternalPath = "/HLT/ScoutingOnline/Muons/Properties")

ScoutingMuonMonitoring = cms.Sequence(ScoutingMuonTagProbeAnalysisNoVtxOnline +
                                      ScoutingMuonTagProbeAnalysisVtxOnline +
                                      ScoutingMuonTriggerAnalysis_DoubleMu +
                                      ScoutingMuonTriggerAnalysis_SingleMu +
                                      ScoutingMuonPropertiesMonitorOnline)
