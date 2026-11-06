import FWCore.ParameterSet.Config as cms

from ..sequences.SimL1TMuonCommonTask_cfi import *
from ..sequences.me0TriggerAllDigiTask_cfi import *
from ..modules.simCscTriggerPrimitiveDigisPhase2_cfi import *
from ..sequences.simMuonGEMPadTask_cfi import *

SimL1TMuonTask = cms.Task(SimL1TMuonCommonTask, me0TriggerAllDigiTask, simCscTriggerPrimitiveDigisPhase2, simMuonGEMPadTask)
