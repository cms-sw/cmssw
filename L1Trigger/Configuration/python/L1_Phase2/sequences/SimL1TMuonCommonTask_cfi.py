import FWCore.ParameterSet.Config as cms

from ..modules.simCscTriggerPrimitiveDigis_cfi import *
from ..modules.simDtTriggerPrimitiveDigis_cfi import *

SimL1TMuonCommonTask = cms.Task(simCscTriggerPrimitiveDigis, simDtTriggerPrimitiveDigis)
