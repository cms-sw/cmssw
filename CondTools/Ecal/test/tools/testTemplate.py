# 
#
# Read from xml and insert into database using
# PopCon 
#
# This is a template, generate real test using
#
# sed 's/RECORDNAME/your-record/g'             testTemplate.py > testyourrecord.py
#
# for example
#

# sed 's/RECORDNAME/EcalADCToGevConstant/g'    testTemplate.py > testEcalADCToGevConstant.py
# sed 's/RECORDNAME/EcalChannelStatus/g'       testTemplate.py > ! testEcalChannelStatus.py
# sed 's/RECORDNAME/EcalGainRatios/g'          testTemplate.py > ! testEcalGainRatios.py
# sed 's/RECORDNAME/EcalIntercalibConstants/g' testTemplate.py > ! testEcalIntercalibConstants.py
# sed 's/RECORDNAME/EcalIntercalibErrors/g'    testTemplate.py > ! testEcalIntercalibErrors.py
# sed 's/RECORDNAME/EcalTBWeights/g'           testTemplate.py > ! testEcalTBWeights.py
# sed 's/RECORDNAME/EcalWeightXtalGroup/g'     testTemplate.py > ! testEcalWeightXtalGroup.py
#
# Stefano Argiro', $Id: testTemplate.py,v 1.1 2010/04/16 08:39:37 depasse Exp $
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
process.CondDBCommon.connect = cms.string('sqlite_file:testRECORDNAME.db')
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
        record = cms.string('RECORDNAMERcd'),
        tag = cms.string('mytest')
         )),
    logconnect= cms.untracked.string('sqlite_file:logtestRECORDNAME.db')                                     
)

process.mytest = cms.EDAnalyzer("RECORDNAMEAnalyzer",
    record = cms.string('RECORDNAMERcd'),
    loggingOn= cms.untracked.bool(True),
    SinceAppendMode=cms.bool(True),
    Source=cms.PSet(
    xmlFile = cms.untracked.string('/tmp/RECORDNAME.xml'),
    since = cms.untracked.int64(3)
    )                            
)

process.p = cms.Path(process.mytest)




