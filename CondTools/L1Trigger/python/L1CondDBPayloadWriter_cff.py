from CondTools.L1Trigger.L1CondDBPayloadWriter_cfi import *

from CondCore.DBCommon.CondDBSetup_cfi import *
outputDB = cms.Service("PoolDBOutputService",
                       CondDBSetup,
                       BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
                       connect = cms.string('sqlite_file:l1config.db'),
                       toPut = cms.VPSet(cms.PSet(
    record = cms.string("L1TriggerKeyListRcd"),
    tag = cms.string("L1TriggerKeyList_IDEAL"))
                                         ))
outputDB.DBParameters.authenticationPath = '.'
