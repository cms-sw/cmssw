import FWCore.ParameterSet.Config as cms

from CondTools.L1Trigger.L1SubsystemParams_cfi import *
L1TriggerKeyDummy = cms.ESProducer("L1TriggerKeyDummyProd",
    objectKeys = cms.VPSet(),
    tscKey = cms.string('dummy'),
    csctfKey = cms.string('dummy'),
    dttfKey = cms.string('dummy'),
    rpcKey = cms.string('dummy'),
    gmtKey = cms.string('dummy'),
    rctKey = cms.string('dummy'),
    gctKey = cms.string('dummy'),
    gtKey = cms.string('dummy'),
    tsp0Key = cms.string('dummy'),
    label = cms.string('')
)

L1TriggerKeyDummy.objectKeys.extend(L1SubsystemParams.recordInfo)

