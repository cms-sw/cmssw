import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_phase2_GE0_cff import phase2_GE0

from L1Trigger.L1TGEM.simMuonME0PadDigis_cfi import *
from L1Trigger.L1TGEM.me0TriggerDigis_cfi import *
from L1Trigger.L1TGEM.me0TriggerPseudoDigis_cff import *

me0TriggerRealDigiTask = cms.Task(simMuonME0PadDigis, me0TriggerDigis)
me0TriggerAllDigiTask = cms.Task(me0TriggerRealDigiTask, me0TriggerPseudoDigiTask)
ge0TriggerAllDigiTask = cms.Task(ge0TriggerPseudoDigiTask)

# in scenarios with GE0, remove the ME0 pseudo digis
phase2_GE0.toReplaceWith(me0TriggerAllDigiTask, ge0TriggerAllDigiTask)
