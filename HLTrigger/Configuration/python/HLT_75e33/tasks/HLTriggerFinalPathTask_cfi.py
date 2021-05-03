import FWCore.ParameterSet.Config as cms

from ..modules.hltFEDSelector_cfi import *
from ..modules.hltGtStage2Digis_cfi import *
from ..modules.hltScalersRawToDigi_cfi import *
from ..modules.hltTriggerSummaryAOD_cfi import *
from ..modules.hltTriggerSummaryRAW_cfi import *

HLTriggerFinalPathTask = cms.Task(hltFEDSelector, hltGtStage2Digis, hltScalersRawToDigi, hltTriggerSummaryAOD, hltTriggerSummaryRAW)
