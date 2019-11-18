import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TGEM.simMuonME0PadDigis_cfi import *
from L1Trigger.L1TGEM.me0TriggerDigis_cfi import *
from L1Trigger.L1TGEM.me0TriggerPseudoDigis_cff import *

me0TriggerRealDigiTask = cms.Task(simMuonME0PadDigis, me0TriggerDigis)
me0TriggerAllDigiTask = cms.Task(me0TriggerRealDigiTask, me0TriggerPseudoDigiTask)
