import FWCore.ParameterSet.Config as cms

from HLTriggerOffline.Scouting.HLTScoutingPi0Monitor_cfi import ScoutingPi0Monitor  as _ScoutingPi0Monitor
ScoutingPi0MonitorOnline = _ScoutingPi0Monitor.clone(OutputInternalPath = cms.string('HLT/ScoutingOnline/PiZero'))
