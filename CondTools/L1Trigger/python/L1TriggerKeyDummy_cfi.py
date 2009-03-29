import FWCore.ParameterSet.Config as cms

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

from CondTools.L1Trigger.L1SubsystemParams_cfi import initL1Subsystems
initL1Subsystems( tagBase = 'IDEAL', objectKey = 'dummy' )
L1TriggerKeyDummy.objectKeys.extend(initL1Subsystems.params.recordInfo)
