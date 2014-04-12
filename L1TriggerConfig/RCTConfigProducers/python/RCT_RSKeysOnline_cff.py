import FWCore.ParameterSet.Config as cms
from CondTools.L1Trigger.L1TriggerKeyDummy_cff import *
L1TriggerKeyDummy.objectKeys = cms.VPSet()
L1TriggerKeyDummy.label = cms.string('SubsystemKeysOnly')
from L1TriggerConfig.RCTConfigProducers.RCT_RSKeysOnline_cfi import *
RCT_RSKeysOnline.subsystemLabel = ''
