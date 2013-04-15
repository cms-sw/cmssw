import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.HLTTauDQMOffline_cfi import *

HLTTauPostAnalysis_PFTaus = cms.EDAnalyzer("HLTTauPostProcessor",
    HLTProcessName  = hltTauOfflineMonitor_PFTaus.HLTProcessName,
    DQMBaseFolder   = hltTauOfflineMonitor_PFTaus.DQMBaseFolder,
    Setup           = hltTauOfflineMonitor_PFTaus.MonitorSetup,
)

HLTTauPostAnalysis_Inclusive = cms.EDAnalyzer("HLTTauPostProcessor",
    HLTProcessName  = hltTauOfflineMonitor_Inclusive.HLTProcessName,
    DQMBaseFolder   = hltTauOfflineMonitor_Inclusive.DQMBaseFolder,
    Setup           = hltTauOfflineMonitor_Inclusive.MonitorSetup,
)

HLTTauPostSeq = cms.Sequence(HLTTauPostAnalysis_PFTaus*HLTTauPostAnalysis_Inclusive)
