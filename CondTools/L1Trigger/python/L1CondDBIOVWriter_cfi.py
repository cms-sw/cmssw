import FWCore.ParameterSet.Config as cms

from CondTools.L1Trigger.L1SubsystemParams_cfi import *
L1CondDBIOVWriter = cms.EDFilter("L1CondDBIOVWriter",
    offlineDB = cms.string('sqlite_file:l1config.db'),
    toPut = cms.VPSet(),
    offlineAuthentication = cms.string(''),
    L1TriggerKeyTag = cms.string('L1TriggerKey_IDEAL')
)

L1CondDBIOVWriter.toPut.extend(L1SubsystemParams.recordInfo)

