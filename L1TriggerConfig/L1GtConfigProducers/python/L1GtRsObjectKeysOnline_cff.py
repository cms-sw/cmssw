# cff file to produce the L1 GT keys for records managed via RUN SETTINGS

import FWCore.ParameterSet.Config as cms

from CondTools.L1Trigger.L1TriggerKeyDummy_cff import *
L1TriggerKeyDummy.objectKeys = cms.VPSet()
L1TriggerKeyDummy.label = cms.string('SubsystemKeysOnly')

from L1TriggerConfig.L1GtConfigProducers.l1GtRsObjectKeysOnline_cfi import *
