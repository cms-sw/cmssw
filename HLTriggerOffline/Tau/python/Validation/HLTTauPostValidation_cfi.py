import FWCore.ParameterSet.Config as cms

from HLTriggerOffline.Tau.Validation.HLTTauValidation_cfi import *

HLTTauValPostAnalysis_MC = cms.EDAnalyzer("HLTTauPostProcessor",
    SourceModule    = hltTauValIdealMonitorMC.ModuleName,
    HLTProcessName  = hltTauValIdealMonitorMC.HLTProcessName,
    DQMBaseFolder   = hltTauValIdealMonitorMC.DQMBaseFolder,
    Setup           = hltTauValIdealMonitorMC.MonitorSetup,
)

HLTTauValPostAnalysis_PF = cms.EDAnalyzer("HLTTauPostProcessor",
    SourceModule    = hltTauValIdealMonitorPF.ModuleName,
    HLTProcessName  = hltTauValIdealMonitorPF.HLTProcessName,
    DQMBaseFolder   = hltTauValIdealMonitorPF.DQMBaseFolder,
    Setup           = hltTauValIdealMonitorPF.MonitorSetup,
)

HLTTauPostVal = cms.Sequence(HLTTauValPostAnalysis_MC+HLTTauValPostAnalysis_PF)
