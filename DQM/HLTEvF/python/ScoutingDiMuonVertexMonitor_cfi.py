import FWCore.ParameterSet.Config as cms

from HLTriggerOffline.Scouting.HLTScoutingDiMuonVertexMonitor_cfi import ScoutingDiMuonVertexMonitor as _ScoutingDiMuonVertexMonitor
ScoutingDiMuonVertexMonitorOnline = _ScoutingDiMuonVertexMonitor.clone(FolderName = cms.string('HLT/ScoutingOnline/DiMuon'))
