'''
DQMEDAnalyzer to read scouting muon collection and scouting vertex collection used in 
ScoutingMuonTagProbeAnalyzer.cc. 

Author: Javier Garcia de Castro, email:javigdc@bu.edu
'''

import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

#Collection to read in ScoutingMuonTagProbeAnlyzer.cc
ScoutingMuonTagProbeAnalysisNoVtx = DQMEDAnalyzer('ScoutingMuonTagProbeAnalyzer',
    OutputInternalPath = cms.string('/HLT/ScoutingOffline/Muons/NoVtx'),
    ScoutingMuonCollection = cms.InputTag('hltScoutingMuonPackerNoVtx'),
    ScoutingVtxCollection = cms.InputTag('hltScoutingMuonPackerNoVtx','displacedVtx','HLT'),
    runWithoutVertex = cms.bool(False)
)

#Name given to add to the sequence in test/runScoutingMonitoringDQM_muonOnly_cfg.py
scoutingMonitoringTagProbeMuonNoVtx = cms.Sequence(ScoutingMuonTagProbeAnalysisNoVtx)

#Clone for the other collection and change only the necessary inputs
ScoutingMuonTagProbeAnalysisVtx = ScoutingMuonTagProbeAnalysisNoVtx.clone()
ScoutingMuonTagProbeAnalysisVtx.OutputInternalPath = cms.string('/HLT/ScoutingOffline/Muons/Vtx') 
ScoutingMuonTagProbeAnalysisVtx.ScoutingMuonCollection = cms.InputTag('hltScoutingMuonPackerVtx')
ScoutingMuonTagProbeAnalysisVtx.ScoutingVtxCollection = cms.InputTag('hltScoutingMuonPackerVtx','displacedVtx','HLT')
ScoutingMuonTagProbeAnalysisVtx.runWithoutVertex = cms.bool(True)

#Name given to add to the sequence in test/runScoutingMonitoringDQM_muonOnly_cfg.py
scoutingMonitoringTagProbeMuonVtx = cms.Sequence(ScoutingMuonTagProbeAnalysisVtx)
