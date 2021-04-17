# 
#
# Read from xml and insert into database using
# PopCon 
#
# This is a template, generate real test using
#
# sed 's/EcalTBWeights/your-record/g' testTemplate.py > testyourrecord.py
#
# Stefano Argiro', $Id: testEcalTBWeights.py,v 1.1 2008/11/14 15:46:03 argiro Exp $
#
#

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('INFO')
    )
)

process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = cms.string('sqlite_file:testEcalTBWeights.db')
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
        record = cms.string('EcalTBWeightsRcd'),
        tag = cms.string('EcalTBWeights_EBEE_BeamComm09_offline')
         )),
    logconnect= cms.untracked.string('sqlite_file:logtestEcalTBWeights.db')                                     
)

#    xmlFile = cms.untracked.string('CMS_Ecal_Weights_V0.xml'),

process.mytest = cms.EDAnalyzer("EcalTBWeightsAnalyzer",
    record = cms.string('EcalTBWeightsRcd'),
    loggingOn= cms.untracked.bool(True),
    SinceAppendMode=cms.bool(True),
    Source=cms.PSet(
    xmlFile = cms.untracked.string('EcalWeightsBasedOnEcalSimAlgoV02-01-04_0.xml'),
    since = cms.untracked.int64(1)
    )                            
)

process.p = cms.Path(process.mytest)




