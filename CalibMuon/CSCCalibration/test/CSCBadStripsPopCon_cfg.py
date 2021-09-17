# The following comments couldn't be translated into the new config version:

#change the firstRun if you want a different IOV

# eg to write payload to the oracle database 
#   replace CondDBCommon.connect = "oracle://cms_orcoff_int2r/CMS_COND_CSC"
# Database output service

import FWCore.ParameterSet.Config as cms

process = cms.Process("ProcessOne")
#PopCon config
process.load("CondCore.DBCommon.CondDBCommon_cfi")

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
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    timetype = cms.string('runnumber'),
    interval = cms.uint64(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    logconnect = cms.untracked.string('sqlite_file:badStripslog.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('CSCBadStripsRcd'),
        tag = cms.string('CSCBadStrips')
    ))
)

process.WriteGainsWithPopCon = cms.EDAnalyzer("CSCBadStripsPopConAnalyzer",
    SinceAppendMode = cms.bool(True),
    record = cms.string('CSCBadStripsRcd'),
    loggingOn = cms.untracked.bool(True),
    Source = cms.PSet(

    )
)

process.p = cms.Path(process.WriteGainsWithPopCon)
process.CondDBCommon.connect = 'sqlite_file:BadStrips.db'


