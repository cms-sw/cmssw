#!/bin/csh
cmsenv

if ("$5" == "db") then
    echo "Loading to ORCON!"
    set connectstring = oracle://cms_orcon_prod/CMS_COND_31X_HCAL
    set authPath = /nfshome0/popcondev/conddb
    set logstring = oracle://cms_orcon_prod/CMS_COND_31X_POPCONLOG
else
     echo "Loading to sqlite_file:test.db"
     set connectstring = sqlite_file:test.db
     set logstring = sqlite_file:log.db
     set authPath = 
endif
    echo "connectstring = $connectstring"


cat >! temp_write_cfg.py <<%

import FWCore.ParameterSet.Config as cms

process = cms.Process("ProcessOne")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.DBParameters.authenticationPath = cms.untracked.string('$authPath' )
process.CondDBCommon.connect = cms.string('$connectstring')

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    interval = cms.uint64(1)
)

process.es_ascii = cms.ESSource("HcalTextCalibrations",
    input = cms.VPSet(cms.PSet(
        object = cms.string('$1'),
        file = cms.FileInPath('$2')
    ))
)

process.prefer("es_ascii")

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    logconnect = cms.untracked.string('$logstring'),
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('Hcal$1Rcd'),
        tag = cms.string("$3")
    ))
)
process.WriteInDB = cms.EDAnalyzer("Hcal$1PopConAnalyzer",
    SinceAppendMode = cms.bool(True),
    record = cms.string('Hcal$1Rcd'),
    loggingOn = cms.untracked.bool(True),
   Source = cms.PSet(
        IOVRun = cms.untracked.uint32($4)
    )
)

process.p = cms.Path(process.WriteInDB)
%
cmsRun temp_write_cfg.py
rm temp_write_cfg.py

