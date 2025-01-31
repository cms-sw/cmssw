import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

ScoutingMuonTagProbeAnalysisNoVtx = DQMEDAnalyzer('ScoutingMuonTagProbeAnalyzer',

    OutputInternalPath = cms.string('/HLT/ScoutingOffline/Muons/NoVtx'),
    MuonCollection = cms.InputTag('slimmedMuons'),
    ScoutingMuonCollection = cms.InputTag('hltScoutingMuonPackerNoVtx'),
    ScoutingVtxCollection = cms.InputTag('hltScoutingMuonPackerNoVtx','displacedVtx','HLT'),
    runWithoutVertex = cms.bool(False)
)

scoutingMonitoringTagProbeMuonNoVtx = cms.Sequence(ScoutingMuonTagProbeAnalysisNoVtx)

ScoutingMuonTagProbeAnalysisVtx = ScoutingMuonTagProbeAnalysisNoVtx.clone()
ScoutingMuonTagProbeAnalysisVtx.OutputInternalPath = cms.string('/HLT/ScoutingOffline/Muons/Vtx') 
ScoutingMuonTagProbeAnalysisVtx.ScoutingMuonCollection = cms.InputTag('hltScoutingMuonPackerVtx')
ScoutingMuonTagProbeAnalysisVtx.ScoutingVtxCollection = cms.InputTag('hltScoutingMuonPackerVtx','displacedVtx','HLT')
ScoutingMuonTagProbeAnalysisVtx.runWithoutVertex = cms.bool(True)


scoutingMonitoringTagProbeMuonVtx = cms.Sequence(ScoutingMuonTagProbeAnalysisVtx)
