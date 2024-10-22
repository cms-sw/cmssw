import FWCore.ParameterSet.Config as cms

from ..modules.hltBoolFalse_cfi import *
from ..modules.hltTriggerSummaryAOD_cfi import *
from ..modules.hltTriggerSummaryRAW_cfi import *

HLTriggerFinalPath = cms.Path(hltTriggerSummaryAOD+hltTriggerSummaryRAW+hltBoolFalse)
