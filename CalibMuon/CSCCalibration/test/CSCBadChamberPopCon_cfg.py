# The following comments couldn't be translated into the new config version:

#change the firstRun if you want a different IOV

# eg to write payload to the oracle database 
#   replace CondDBCommon.connect = "oracle://cms_orcoff_prep/CMS_COND_CSC"

# Database output service

import FWCore.ParameterSet.Config as cms

process = cms.Process("ProcessOne")
#PopCon config
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = 'sqlite_file:BadChambers_5ME42.db'
process.CondDBCommon.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'
#process.CondDBCommon.DBParameters.messageLevel = cms.untracked.int32(3)
process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("EmptyIOVSource",
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    timetype = cms.string('runnumber'),
    interval = cms.uint64(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    logconnect = cms.untracked.string('sqlite_file:badChamberslog.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('CSCBadChambersRcd'),
        tag = cms.string('CSCBadChambers_none_FiveLiveME42')
    ))
)

process.WriteBadChambersWithPopCon = cms.EDAnalyzer("CSCBadChambersPopConAnalyzer",
    SinceAppendMode = cms.bool(True),
    record = cms.string('CSCBadChambersRcd'),
    loggingOn = cms.untracked.bool(True),
    Source = cms.PSet(

    )
)

process.p = cms.Path(process.WriteBadChambersWithPopCon)



