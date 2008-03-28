import FWCore.ParameterSet.Config as cms

from CondTools.L1Trigger.L1SubsystemParams_cfi import *
L1TriggerKeyDummy = cms.ESProducer("L1TriggerKeyDummyProd",
    subsystemKeys = cms.VPSet(),
    tscKey = cms.string('dummy')
)

L1TriggerKeyDummy.subsystemKeys.append(L1SubsystemParams.recordInfo)

