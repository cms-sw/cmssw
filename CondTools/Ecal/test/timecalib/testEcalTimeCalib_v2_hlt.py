# 
#
# Read from xml and insert into database using
# PopCon 
#
# This is a template, generate real test using
#
# sed 's/EcalGainRatios/your-record/g' testTemplate.py > testyourrecord.py
#
# Stefano Argiro', $Id: testEcalTimeCalib_v2_hlt.py,v 1.1 2010/04/15 12:31:35 depasse Exp $
#
#

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.MessageLogger=cms.Service("MessageLogger",
                              destinations=cms.untracked.vstring("cout"),
                              cout=cms.untracked.PSet(
                              )
)

process.load("CondCore.DBCommon.CondDBCommon_cfi")
#process.CondDBCommon.connect = cms.string('oracle://cms_orcon_prod/CMS_COND_31X_ECAL')
process.CondDBCommon.connect = cms.string('sqlite_file:testEcalTimeCalib.db')
process.CondDBCommon.DBParameters.authenticationPath = cms.untracked.string('/nfshome0/popcondev/conddb')

process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue  = cms.uint64(1),
    interval = cms.uint64(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('EcalTimeCalibConstantsRcd'),
        tag = cms.string('EcalTimeCalibConstants_v2_hlt')
         )),
    logconnect= cms.untracked.string('sqlite_file:logtestEcalTimeCalib.db')                                     
)

process.mytest = cms.EDAnalyzer("EcalTimeCalibConstantsAnalyzer",
    record = cms.string('EcalTimeCalibConstantsRcd'),
    loggingOn= cms.untracked.bool(True),
    SinceAppendMode=cms.bool(True),
    Source=cms.PSet(
    xmlFile = cms.untracked.string('EcalTimeCalibConstants.xml'),
    since = cms.untracked.int64(130837)
    )                            
)

process.p = cms.Path(process.mytest)




