# $Id$
#
# Author: Stefano Argiro'
#
# Script to load ECAL conditions to DB using PopCon
# Intended to be used with the drop-box mechanism, where an XML file
# containing Ecal conditions is sent to DB
#

import FWCore.ParameterSet.Config as cms
from   elementtree.ElementTree import parse
import sys,os,getopt


##############################################################################
# User Defined parameters to be modified
# This will not be needed as soon as we will be able to pass arguments
# to cmsRun
#


#file from which we are reading the conditions to load
filename_  = '/tmp/EcalChannelStatus.xml'

#record type, without 'Rcd'
record_    = 'EcalChannelStatus'

#db parameters, dbconnect can be 'sqlite_file:bla' 
dbconnect_    = 'sqlite_file:testEcalChannelStatus.db'

#
# End User Defined parameters
#############################################################################

def usage():

   print "Usage: cmsRun loadConditions.py "
   sys.exit(2)


def readTagAndSince(filename, headertag='EcalCondHeader'):
    '''Read tag and since from EcalCondHeader in XML file '''
    root   = parse(filename).getroot()
    header = root.find(headertag)
    since  = header.findtext('since') 
    tag    = header.findtext('tag')     

    return tag,since


tag_ , since_ = readTagAndSince(filename_)

#which analyzer to use for each record name
analyzer_ =   {'EcalGainRatios':'EcalGainRatiosAnalyzer',             \
               'EcalADCToGeVConstant':'EcalADCToGeVConstantAnalyzer', \
               'EcalWeightXtalGroups':'EcalWeightGroupAnalyzer',      \
               'EcalChannelStatus':'EcalChannelStatusHandler',        \
               'EcalChannelStatus':'EcalChannelStatusAnalyzer',       \
               'EcalTBWeights':'EcalTBWeightsAnalyzer',               \
               'EcalIntercalibConstants':'EcalIntercalibConstantsAnalyzer', \
               'EcalIntercalibErrors':'EcalIntercalibErrorsAnalyzer'}




## def main():

##    try:
##       opts, args = getopt.getopt(sys.argv[1:], "f:d", ["file=","dryrun"])

##    except getopt.GetoptError:
##       #print help information and exit
##       usage()
##       sys.exit(2)

##    file  = ''
##    dryrun= False
   
##    for opt, arg in opts:   
##      if opt in ("-f", "--file"):
##          file = arg
##          if (not os.path.exists(file)) :
##             print sys.argv[0]+" File not found: "+file
##             sys.exit(2)

##    if file=='':
##        usage()
##        exit(2)
   
##    tag,since = readTagAndSince(file)
##    print tag,since


process = cms.Process("LoadEcalConditions")

process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue  = cms.uint64(1),
    interval = cms.uint64(1)
)


process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = cms.string(dbconnect_)
process.CondDBCommon.DBParameters.authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string(record_+'Rcd'),
        tag = cms.string(tag_)
         )),
        logconnect= cms.untracked.string('sqlite_file:log.db')                                  
)


process.popconAnalyzer = cms.EDAnalyzer(analyzer_[record_],
    record = cms.string(record_+'Rcd'),
    loggingOn= cms.untracked.bool(True),
    SinceAppendMode=cms.bool(True),
    Source=cms.PSet(
    xmlFile = cms.untracked.string(filename_),
    since = cms.untracked.int64(int(since_)) #python will make the int as
                                             #long as needed
    )                            
)    


process.p = cms.Path(process.popconAnalyzer)
