# The following comments couldn't be translated into the new config version:

# eg to write payload to the oracle database 
#   replace CondDBCommon.connect = "oracle://cms_orcoff_int2r/CMS_COND_CSC"
# Database output service

import FWCore.ParameterSet.Config as cms

process = cms.Process("ProcessOne")
#PopCon config
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = cms.string("sqlite_file:CSCChamberIndexValues_20X.db")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        enable = cms.untracked.bool(True)
    )
)

process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(1),
    timetype = cms.string('runnumber'),
    #change the firstRun if you want a different IOV
    firstValue = cms.uint64(1),
    interval = cms.uint64(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    logconnect = cms.untracked.string('sqlite_file:chamberindex20X.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('CSCChamberIndexRcd'),
        tag = cms.string('CSCChamberIndex')
    ))
)

process.WriteInDB = cms.EDAnalyzer("CSCChamberIndexPopConAnalyzer",
    SinceAppendMode = cms.bool(True),
    record = cms.string('CSCChamberIndexRcd'),
    loggingOn = cms.untracked.bool(True),
    Source = cms.PSet(

    )
)

process.p = cms.Path(process.WriteInDB)


