import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.HLTTauDQMOffline_cfi import *

HLTTauPostAnalysis_PFTaus = cms.EDAnalyzer("HLTTauPostProcessor",
    DQMBaseFolder   = hltTauOfflineMonitor_PFTaus.DQMBaseFolder,
    L1Plotter       = hltTauOfflineMonitor_PFTaus.L1Plotter,
    PathSummaryPlotter = hltTauOfflineMonitor_PFTaus.PathSummaryPlotter,
)

HLTTauPostAnalysis_Inclusive = cms.EDAnalyzer("HLTTauPostProcessor",
    DQMBaseFolder   = hltTauOfflineMonitor_Inclusive.DQMBaseFolder,
    L1Plotter       = hltTauOfflineMonitor_Inclusive.L1Plotter,
    PathSummaryPlotter = hltTauOfflineMonitor_Inclusive.PathSummaryPlotter,
)

HLTTauPostSeq = cms.Sequence(HLTTauPostAnalysis_PFTaus*HLTTauPostAnalysis_Inclusive)
