import FWCore.ParameterSet.Config as cms
from CondTools.L1Trigger.L1TriggerKeyDummy_cff import *
L1TriggerKeyDummy.objectKeys = cms.VPSet()
L1TriggerKeyDummy.label = cms.string('SubsystemKeysOnly')
from L1TriggerConfig.GMTConfigProducers.L1MuGMTRSKeysOnline_cfi import *
L1MuGMTRSKeysOnline.subsystemLabel = ''
# foo bar baz
# DImyZTWmY7HlB
# bLkOYMVM9LYh8
