import FWCore.ParameterSet.Config as cms

from HLTriggerOffline.Scouting.HLTScoutingDileptonMonitor_cfi import ScoutingDileptonMonitor as _ScoutingDileptonMonitor
ScoutingDileptonMonitorOnline = _ScoutingDileptonMonitor.clone(OutputInternalPath = cms.string('HLT/ScoutingOnline/DiLepton'))
