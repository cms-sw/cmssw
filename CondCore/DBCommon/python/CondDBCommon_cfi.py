print " ################################################################### "  
print " # WARNING: this module is deprecated.                             # "                                                                              
print " # Please use CondCore.CondDB.CondDB_cfi.py                        # "                                                                             
print " ################################################################### "

import FWCore.ParameterSet.Config as cms

CondDBCommon = cms.PSet(
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string(''),
        authenticationSystem = cms.untracked.int32(0),
        connectionRetrialPeriod = cms.untracked.int32(10),
        idleConnectionCleanupPeriod = cms.untracked.int32(10),
        messageLevel = cms.untracked.int32(0),
        enablePoolAutomaticCleanUp = cms.untracked.bool(False),
        enableConnectionSharing = cms.untracked.bool(True),
        connectionRetrialTimeOut = cms.untracked.int32(60),
        connectionTimeOut = cms.untracked.int32(60),
        enableReadOnlySessionOnUpdateConnection = cms.untracked.bool(False)
    ),
    connect = cms.string('protocol://db/schema') ##db/schema"

)

