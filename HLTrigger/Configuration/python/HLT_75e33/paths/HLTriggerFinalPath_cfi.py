import FWCore.ParameterSet.Config as cms

from ..modules.hltBoolFalse_cfi import *
from ..tasks.HLTriggerFinalPathTask_cfi import *

HLTriggerFinalPath = cms.Path(hltBoolFalse, HLTriggerFinalPathTask)
