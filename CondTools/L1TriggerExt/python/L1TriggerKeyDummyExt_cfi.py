import FWCore.ParameterSet.Config as cms

L1TriggerKeyDummyExt = cms.ESProducer("L1TriggerKeyDummyProdExt",
    objectKeys = cms.VPSet(),
    tscKey = cms.string('dummy'),
    uGTKey = cms.string('dummy'),
    label = cms.string('')
)

from CondTools.L1TriggerExt.L1UniformTagsExt_cfi import initL1UniformTagsExt
initL1UniformTagsExt( tagBase = 'IDEAL' )
from CondTools.L1TriggerExt.L1SubsystemParamsExt_cfi import initL1SubsystemsExt
initL1SubsystemsExt( tagBaseVec = initL1UniformTagsExt.tagBaseVec, objectKey = 'dummy' )
L1TriggerKeyDummyExt.objectKeys.extend(initL1SubsystemsExt.params.recordInfo)
