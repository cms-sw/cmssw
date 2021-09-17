# 
#
# Read from xml and insert into database using
# PopCon 
#
# This is a template, generate real test using
#
# sed 's/EcalWeightXtalGroup/your-record/g' testTemplate.py > testyourrecord.py
#
# Stefano Argiro', $Id: testEcalWeightXtalGroup.py,v 1.1 2008/11/14 15:46:03 argiro Exp $
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

process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = cms.string('sqlite_file:testEcalWeightXtalGroups.db')
#process.CondDBCommon.connect = cms.string('oracle://cms_orcon_prod/CMS_COND_31X_ECAL')
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
        record = cms.string('EcalWeightXtalGroupsRcd'),
        tag = cms.string('EcalWeightXtalGroups_EBEE_mc')
         )),
    logconnect= cms.untracked.string('sqlite_file:logtestEcalWeightXtalGroups.db')
)

process.mytest = cms.EDAnalyzer("EcalWeightGroupAnalyzer",
    record = cms.string('EcalWeightXtalGroupsRcd'),
    loggingOn= cms.untracked.bool(True),
    SinceAppendMode=cms.bool(True),
    Source=cms.PSet(
      xmlFile = cms.untracked.string('CMS_Ecal_XtalGroupIDs_slc5.xml'),
      since = cms.untracked.int64(1)
    )                            
)

process.p = cms.Path(process.mytest)




