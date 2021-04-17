import FWCore.ParameterSet.Config as cms

from DQMOffline.Muon.gemEfficiencyAnalyzer_cfi import *

gemSources = cms.Sequence(
    gemEfficiencyAnalyzerTightGlbSeq *
    gemEfficiencyAnalyzerStaSeq
)
