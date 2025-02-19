# $Id: loadConditions.py,v 1.5 2009/05/13 13:48:05 argiro Exp $
#
# Author: Stefano Argiro'
#
# Script to load ECAL conditions to DB using PopCon
# Intended to be used with the drop-box mechanism, where an XML file
# containing Ecal conditions is sent to DB
#

import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
from   elementtree.ElementTree import parse
import sys,os




def usage():

   print "Usage: cmsRun loadConditions.py file=FILENAME record=RECORD db=CONNECTSTRING"
   print "   file=FILE"
   print "       specify xml file to load to DB"
   print
   print "   record=RECORD"
   print "       specify record to be loaded (EcalChannelStatus, etc)"
   print 
   print "   db=CONNECTSTRING"
   print "       specify connection string, e.g. sqlite_file=file.db"
   print 


usage()


options = VarParsing.VarParsing ()
options.register ('file',
                  "", # default value
                  VarParsing.VarParsing.multiplicity.singleton, 
                  VarParsing.VarParsing.varType.string,     
                  "xml file to load")

options.register ('record',
                  "", # default value
                  VarParsing.VarParsing.multiplicity.singleton, 
                  VarParsing.VarParsing.varType.string,     
                  "record type to load")

options.register ('db',
                  "", # default value
                  VarParsing.VarParsing.multiplicity.singleton, 
                  VarParsing.VarParsing.varType.string,     
                  "db connection string")

options.parseArguments()


def readTagAndSince(filename, headertag='EcalCondHeader'):
    '''Read tag and since from EcalCondHeader in XML file '''
    root   = parse(filename).getroot()
    header = root.find(headertag)
    since  = header.findtext('since') 
    tag    = header.findtext('tag')     

    return tag,since


tag_ , since_ = readTagAndSince(options.file)

#which analyzer to use for each record name
analyzer_ =   {'EcalGainRatios':'EcalGainRatiosAnalyzer',             \
               'EcalADCToGeVConstant':'EcalADCToGeVConstantAnalyzer', \
               'EcalWeightXtalGroups':'EcalWeightGroupAnalyzer',      \
               'EcalChannelStatus':'EcalChannelStatusHandler',        \
               'EcalChannelStatus':'EcalChannelStatusAnalyzer',       \
               'EcalTBWeights':'EcalTBWeightsAnalyzer',               \
               'EcalIntercalibConstants':'EcalIntercalibConstantsAnalyzer', \
               'EcalIntercalibConstantsMC':'EcalIntercalibConstantsMCAnalyzer', \ 
               'EcalIntercalibErrors':'EcalIntercalibErrorsAnalyzer'
               }



process = cms.Process("LoadEcalConditions")

process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue  = cms.uint64(1),
    interval = cms.uint64(1)
)


process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = cms.string(options.db)
process.CondDBCommon.DBParameters.authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string(options.record+'Rcd'),
        tag = cms.string(tag_)
         )),
        logconnect= cms.untracked.string('sqlite_file:log.db')                                  
)


process.popconAnalyzer = cms.EDAnalyzer(analyzer_[options.record],
    record = cms.string(options.record+'Rcd'),
    loggingOn= cms.untracked.bool(True),
    SinceAppendMode=cms.bool(True),
    Source=cms.PSet(
    xmlFile = cms.untracked.string(options.file),
    since = cms.untracked.int64(int(since_)) #python will make the int as
                                             #long as needed
    )                            
)    


process.p = cms.Path(process.popconAnalyzer)


