import FWCore.ParameterSet.Config as cms

from HLTriggerOffline.Tau.Validation.HLTTauValidation_cfi import *
import DQMOffline.Trigger.HLTTauPostProcessor_cfi as postProcessor

HLTTauValPostAnalysis_MC = postProcessor.makePFTauAnalyzer(hltTauValIdealMonitorMC)
HLTTauValPostAnalysis_PF = postProcessor.makePFTauAnalyzer(hltTauValIdealMonitorPF)
HLTTauPostVal = cms.Sequence(HLTTauValPostAnalysis_MC+HLTTauValPostAnalysis_PF)
