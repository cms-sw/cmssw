import FWCore.ParameterSet.Config as cms

import copy
from L1Trigger.GlobalTriggerAnalyzer.l1GtTrigReport_cfi import *
l1gtTrigReport = copy.deepcopy(l1GtTrigReport)
import copy
from HLTrigger.HLTanalyzers.hlTrigReport_cfi import *
hltTrigReport = copy.deepcopy(hlTrigReport)
import copy
from HLTrigger.HLTcore.triggerSummaryProducerAOD_cfi import *
triggerSummaryAOD = copy.deepcopy(triggerSummaryProducerAOD)
import copy
from HLTrigger.HLTcore.triggerSummaryProducerRAW_cfi import *
triggerSummaryRAW = copy.deepcopy(triggerSummaryProducerRAW)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
triggerSummaryRAWprescaler = copy.deepcopy(hltPrescaler)
import copy
from HLTrigger.HLTfilters.hltBool_cfi import *
boolFinal = copy.deepcopy(hltBool)
options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)
HLTEndpath1 = cms.EndPath(l1gtTrigReport+hltTrigReport)
triggerFinalPath = cms.Sequence(triggerSummaryAOD+triggerSummaryRAWprescaler+triggerSummaryRAW+boolFinal)
l1gtTrigReport.L1GtRecordInputTag = 'gtDigis'
boolFinal.result = False

