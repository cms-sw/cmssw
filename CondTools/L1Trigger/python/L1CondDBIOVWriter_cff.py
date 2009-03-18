from CondTools.L1Trigger.L1CondDBIOVWriter_cfi import *

from CondCore.DBCommon.CondDBSetup_cfi import *
outputDB = cms.Service("PoolDBOutputService",
                       CondDBSetup,
                       BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
                       connect = cms.string('sqlite_file:l1config.db'),
                       toPut = cms.VPSet(cms.PSet(
    record = cms.string("L1TriggerKeyRcd"),
    tag = cms.string("L1TriggerKey_IDEAL"))
                                         ))
outputDB.DBParameters.authenticationPath = '.'

from CondTools.L1Trigger.L1SubsystemParams_cfi import *
outputDB.toPut.extend(L1SubsystemParams.recordInfo)
