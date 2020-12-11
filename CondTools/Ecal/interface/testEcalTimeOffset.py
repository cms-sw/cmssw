# 
#
# Read from xml and insert into database using PopCon 
#
# This is a template, generate real test using
#
# sed 's/EcalGainRatios/your-record/g' testTemplate.py > testyourrecord.py
#
# Stefano Argiro', $Id: testEcalGainRatios.py,v 1.1 2008/11/14 15:46:03 argiro Exp $
#
#

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True)
    )
)
process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = 'sqlite_file:EcalTimeOffsetConstant.db'


process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue  = cms.uint64(1),
    interval = cms.uint64(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDB,
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('EcalTimeOffsetConstantRcd'),
        tag = cms.string('EcalTimeOffsetConstant_204623_minus1ns')
         )),
                                 
)

process.mytest = cms.EDAnalyzer("EcalTimeOffsetConstantAnalyzer",
    record = cms.string('EcalTimeOffsetConstantRcd'),
    loggingOn= cms.untracked.bool(True),
    SinceAppendMode=cms.bool(True),
    Source=cms.PSet(
    xmlFile = cms.untracked.string('EcalTimeOffset_204623_minus1ns.xml'),
    since = cms.untracked.int64(1)
    )
)

process.p = cms.Path(process.mytest)




