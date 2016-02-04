# 
#
# Read from xml and insert into database using
# PopCon 
#
# This is a template, generate real test using
#
# sed 's/EcalGainRatios/your-record/g' testTemplate.py > testyourrecord.py
#
# Stefano Argiro', $Id: testEcalGainRatios.py,v 1.1 2010/04/15 13:17:55 depasse Exp $
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

# pour ecrire dans la DB  
#process.CondDBCommon.connect = cms.string('oracle://cms_orcon_prod/CMS_COND_31X_ECAL')
#process.CondDBCommon.DBParameters.authenticationPath = cms.untracked.string('/nfshome0/popcondev/conddb')

# pour ecrire dans sqlite 
process.CondDBCommon.connect = cms.string('sqlite_file:testEcalGainRatios.db')
process.CondDBCommon.DBParameters.authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')

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
        record = cms.string('EcalGainRatiosRcd'),
        tag = cms.string('EcalGainRatio_TestPulse2009_offline')
         )),
    logconnect= cms.untracked.string('sqlite_file:logtestEcalGainRatios.db')                                     
)

process.mytest = cms.EDAnalyzer("EcalGainRatiosAnalyzer",
    record = cms.string('EcalGainRatiosRcd'),
    loggingOn= cms.untracked.bool(True),
    SinceAppendMode=cms.bool(True),
    Source=cms.PSet(
    xmlFile = cms.untracked.string('EcalGainRatios.xml'),
    since = cms.untracked.int64(1)
    )                            
)

process.p = cms.Path(process.mytest)




