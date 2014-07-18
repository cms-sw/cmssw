import FWCore.ParameterSet.Config as cms

from HLTriggerOffline.Tau.Validation.HLTTauValidation_cfi import *

HLTTauValPostAnalysis_MC = cms.EDAnalyzer("HLTTauPostProcessor",
    DQMBaseFolder   = hltTauValIdealMonitorMC.DQMBaseFolder,
    L1Plotter       = hltTauValIdealMonitorMC.L1Plotter,
    PathSummaryPlotter = hltTauValIdealMonitorMC.L1Plotter,
)

HLTTauValPostAnalysis_PF = cms.EDAnalyzer("HLTTauPostProcessor",
    DQMBaseFolder   = hltTauValIdealMonitorPF.DQMBaseFolder,
    L1Plotter       = hltTauValIdealMonitorPF.L1Plotter,
    PathSummaryPlotter = hltTauValIdealMonitorPF.PathSummaryPlotter,
)

HLTTauPostVal = cms.Sequence(HLTTauValPostAnalysis_MC+HLTTauValPostAnalysis_PF)
