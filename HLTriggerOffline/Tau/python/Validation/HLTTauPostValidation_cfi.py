import FWCore.ParameterSet.Config as cms

from HLTriggerOffline.Tau.Validation.HLTTauValidation_cfi import *
from Validation.RecoTau.hltTauPostProcessor_cff import *
import DQMOffline.Trigger.HLTTauPostProcessor_cfi as postProcessor

(HLTTauValPostAnalysisMC, HLTTauValPostAnalysisMC2) = postProcessor.makePFTauAnalyzer(hltTauValIdealMonitorMC)
(HLTTauValPostAnalysisPF, HLTTauValPostAnalysisPF2) = postProcessor.makePFTauAnalyzer(hltTauValIdealMonitorPF)
(HLTTauValPostAnalysisPN, HLTTauValPostAnalysisPN2) = postProcessor.makePFTauAnalyzer(hltTauValIdealMonitorPNet)
(HLTTauValPostAnalysisTP, HLTTauValPostAnalysisTP2) = postProcessor.makePFTauAnalyzer(hltTauValTagAndProbe)

HLTTauPostVal = cms.Sequence(
    HLTTauValPostAnalysisMC+HLTTauValPostAnalysisMC2+
    HLTTauValPostAnalysisPF+HLTTauValPostAnalysisPF2+
    HLTTauValPostAnalysisPN+HLTTauValPostAnalysisPN2+
    HLTTauValPostAnalysisTP+HLTTauValPostAnalysisTP2
)
HLTTauPostValPhase2 = cms.Sequence(hltTauPostProcessor)

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toReplaceWith(HLTTauPostVal, HLTTauPostValPhase2)