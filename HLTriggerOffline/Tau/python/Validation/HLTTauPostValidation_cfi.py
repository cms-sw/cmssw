import FWCore.ParameterSet.Config as cms

from HLTriggerOffline.Tau.Validation.HLTTauValidation_cfi import *
import DQMOffline.Trigger.HLTTauPostProcessor_cfi as postProcessor

(HLTTauValPostAnalysis_MC, HLTTauValPostAnalysis_MC2) = postProcessor.makePFTauAnalyzer(hltTauValIdealMonitorMC)
(HLTTauValPostAnalysis_PF, HLTTauValPostAnalysis_PF2) = postProcessor.makePFTauAnalyzer(hltTauValIdealMonitorPF)
HLTTauPostVal = cms.Sequence(
    HLTTauValPostAnalysis_MC+HLTTauValPostAnalysis_MC2+
    HLTTauValPostAnalysis_PF+HLTTauValPostAnalysis_PF2
)
