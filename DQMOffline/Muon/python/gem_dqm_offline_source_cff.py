import FWCore.ParameterSet.Config as cms

from DQMOffline.Muon.gemOfflineMonitor_cfi import *
from DQMOffline.Muon.gemEfficiencyAnalyzer_cfi import *

gemSources = cms.Sequence(
    gemOfflineMonitor *
    gemEfficiencyAnalyzerTightSeq *
    gemEfficiencyAnalyzerSTASeq)
