# The following comments couldn't be translated into the new config version:

#change the firstRun if you want a different IOV

# eg to write payload to the oracle database 
#   replace CondDBCommon.connect = "oracle://cms_orcoff_int2r/CMS_COND_CSC"
# Database output service

import FWCore.ParameterSet.Config as cms

process = cms.Process("ProcessOne")
#PopCon config
process.load("CondCore.DBCommon.CondDBCommon_cfi")
#process.CondDBCommon.connect = cms.string("sqlite_file:DBL1TPParameters.db")
process.CondDBCommon.connect = "oracle://cms_orcoff_prep/CMS_COND_CSC"
process.CondDBCommon.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'

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
    logconnect = cms.untracked.string('sqlite_file:L1TPParameterslog.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('CSCDBL1TPParametersRcd'),
        tag = cms.string('CSCDBL1TPParameters_mc')
    ))
)

process.WriteL1TPParametersWithPopCon = cms.EDAnalyzer("CSCDBL1TPParametersPopConAnalyzer",
    SinceAppendMode = cms.bool(True),
    record = cms.string('CSCDBL1TPParametersRcd'),
    loggingOn = cms.untracked.bool(True),
    Source = cms.PSet(

    )
)

process.p = cms.Path(process.WriteL1TPParametersWithPopCon)



