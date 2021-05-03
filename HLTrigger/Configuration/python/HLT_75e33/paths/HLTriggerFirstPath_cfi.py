import FWCore.ParameterSet.Config as cms

from ..modules.hltBoolFalse_cfi import *
from ..modules.hltGetRaw_cfi import *

HLTriggerFirstPath = cms.Path(hltGetRaw+hltBoolFalse)
