import FWCore.ParameterSet.Config as cms

## See DQMOffline/HLTScouting/python/HLTScoutingDqmOffline_cff.py

from HLTriggerOffline.Scouting.ScoutingRecHitAnalyzers_cff import *

ScoutingEBRechitAnalyzerOnline = ScoutingEBRechitAnalyzer.clone()
ScoutingHBHERechitAnalyzerOnline = ScoutingHBHERechitAnalyzer.clone()

ScoutingRecHitsMonitoring = cms.Sequence(ScoutingEBRechitAnalyzerOnline +
                                         ScoutingHBHERechitAnalyzerOnline)
                                         
