import FWCore.ParameterSet.Config as cms

from ..modules.hltTriggerSummaryAOD_cfi import *
from ..modules.hltTriggerSummaryRAW_cfi import *
from ..modules.hltBoolFalse_cfi import *

HLTriggerFinalPath = cms.Path(
    hltTriggerSummaryAOD + 
    hltTriggerSummaryRAW + 
    hltBoolFalse )
# foo bar baz
# m4f8gMGqlMsqj
# GjeEew6DWgbxU
