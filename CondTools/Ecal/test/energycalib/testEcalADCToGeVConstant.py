# 
#
# Read from xml and insert into database using
# PopCon 
#
# This is a template, generate real test using
#
# sed 's/EcalADCToGeVConstant/your-record/g'             testTemplate.py > testyourrecord.py
#
# for example
#

# sed 's/EcalADCToGeVConstant/EcalADCToGevConstant/g'    testTemplate.py > testEcalADCToGevConstant.py
# sed 's/EcalADCToGeVConstant/EcalChannelStatus/g'       testTemplate.py > ! testEcalChannelStatus.py
# sed 's/EcalADCToGeVConstant/EcalGainRatios/g'          testTemplate.py > ! testEcalGainRatios.py
# sed 's/EcalADCToGeVConstant/EcalIntercalibConstants/g' testTemplate.py > ! testEcalIntercalibConstants.py
# sed 's/EcalADCToGeVConstant/EcalIntercalibErrors/g'    testTemplate.py > ! testEcalIntercalibErrors.py
# sed 's/EcalADCToGeVConstant/EcalTBWeights/g'           testTemplate.py > ! testEcalTBWeights.py
# sed 's/EcalADCToGeVConstant/EcalWeightXtalGroup/g'     testTemplate.py > ! testEcalWeightXtalGroup.py
#
# Stefano Argiro', $Id: testEcalADCToGeVConstant.py,v 1.1 2010/04/15 13:09:28 depasse Exp $
#
#

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.MessageLogger=cms.Service("MessageLogger",
                              destinations=cms.untracked.vstring("cout"),
                              cout=cms.untracked.PSet(
                              treshold=cms.untracked.string("INFO")
                              )
)

process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = cms.string('sqlite_file:testEcalADCToGeVConstant.db')
process.CondDBCommon.DBParameters.authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')

process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue  = cms.uint64(2),
    interval = cms.uint64(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('EcalADCToGeVConstantRcd'),
        tag = cms.string('mytest')
         )),
    logconnect= cms.untracked.string('sqlite_file:logtestEcalADCToGeVConstant.db')                                     
)

process.mytest = cms.EDAnalyzer("EcalADCToGeVConstantAnalyzer",
    record = cms.string('EcalADCToGeVConstantRcd'),
    loggingOn= cms.untracked.bool(True),
    SinceAppendMode=cms.bool(True),
    Source=cms.PSet(
    xmlFile = cms.untracked.string('/tmp/EcalADCToGeVConstant.xml'),
    since = cms.untracked.int64(3)
    )                            
)

process.p = cms.Path(process.mytest)




