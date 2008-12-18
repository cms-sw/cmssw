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

#file from which we are reading the 
filename_  = '/tmp/EcalGainRatios.xml'
record_    = 'EcalADCToGeVConstant'

#select db parameters, dbconnect can be 'sqlite_file:bla' 
dbconnect_    = None
tag_ , since_ = ReadTagAndSince(filename_)

#which analyzer to use for each record name
analyzer_ =   {'EcalGainRatios':'EcalGainRatiosAnalyzer',             \
               'EcalADCToGeVConstant':'EcalADCToGeVConstantAnalyzer', \
               'EcalWeightXtalGroups':'EcalWeightGroupAnalyzer',      \
               'EcalChannelStatus':'EcalChannelStatusHandler',        \
               'EcalChannelStatus':'EcalChannelStatusAnalyzer',       \
               'EcalTBWeights','EcalTBWeightsAnalyzer',               \
               'EcalIntercalibConstants','EcalIntercalibConstantsAnalyzer', \
               'EcalIntercalibErrors','EcalIntercalibErrorsAnalyzer'}


def usage():

   print "Usage: "+sys.argv[0]+" Write me !"

   sys.exit(2)


def main():

   try:
      opts, args = getopt.getopt(sys.argv[1:], "f:d", ["file=","dryrun"])

   except getopt.GetoptError:
      #print help information and exit
      usage()
      sys.exit(2)

   file  = ''
   dryrun= False
   
   for opt, arg in opts:   
     if opt in ("-f", "--file"):
         file = arg
         if (not os.path.exists(file)) :
            print sys.argv[0]+" File not found: "+file
            sys.exit(2)

   if file=='':
       usage()
       exit(2)
   
   tag,since = readTagAndSince(file)
   print tag,since



def readTagAndSince(filename, headertag='EcalCondHeader'):
    '''Read tag and since from EcalCondHeader in XML file '''
    root   = parse(filename).getroot()
    header = root.find(headertag)
    since  = header.findtext('since') 
    tag    = header.findtext('tag')     

    return tag,since

process.popconAnalyzer = cms.EDAnalyzer(analyzer_[record],
    record = cms.string(record_+'Rcd'),
    loggingOn= cms.untracked.bool(True),
    SinceAppendMode=cms.bool(True),
    Source=cms.PSet(
    xmlFile = cms.untracked.string(filename_),
    since = cms.untracked.int64(since_)
    )                            
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
)


process.p = cms.Path(process.popconAnalyzer)

if __name__ == "__main__":
    main()
