#!/usr/bin/env python
# Colin Bernet, Dec 2009

import FWCore.ParameterSet.Config as cms
from optparse import OptionParser

import sys,os, re, pprint, imp

def getFiles( dbsOut ):

    pattern = re.compile( '(^/store.*)\n' )
    files = cms.untracked.vstring()
    for line in dbsOut:
        m = pattern.match( line )
        if m:
            files.append( m.group(1) )
    return files
        
def decodeEventInfo( string ):

    spat = '^(\d+):(\d+):(\d+)$'
    pattern = re.compile( spat )
    m = pattern.match( string )
    if m:
        run = m.group(1)
        lumi = m.group(2)
        event = m.group(3)
        return ( run, lumi, event )
    else:
        print string, 'does not match pattern: ', spat
        print 'please specify your event information correctly, like run:lumi:event.'
        sys.exit(2)

def decodeEventsInfo( eventsInfo ): 

    listOfEventInfo = eventsInfo.rsplit(',')
    decodedEventInfos = []
    for eventInfo in listOfEventInfo:
        decodedEventInfos.append( decodeEventInfo( eventInfo ) )
        
    return decodedEventInfos

def buildDBSQuery( decodedEventInfos):
    dbsQuery = 'dbs search --query "find file where dataset=/ExpressPhysics/BeamCommissioning09-Express-v2/FEVT and ( ('
    eventRanges = cms.untracked.VEventRange()
    firstOne = True
    for eventInfo in decodedEventInfos:
        (run, lumi, event) = eventInfo
        orstr = ' or ('
        if firstOne:
            orstr = ''
            firstOne = False
            
        dbsQuery = '%s %s run=%s and lumi=%s )' % (dbsQuery, orstr, run, lumi)
        eventRange =  '%s:%s' % (run, event)
        eventRanges.append( eventRange )

    dbsQuery += ')"'
    return (dbsQuery, eventRanges)

parser = OptionParser()
parser.usage = "%prog <eventsInfo> <cfg>\neventsInfo should be of the form: 'run1:lumi1:event1,run2:lumi2:event2,...'"
parser.add_option("-n", "--negate", action="store_true",
                  dest="negate", default=False,
                  help="create cfg file, but do not cmsRun")

(options,args) = parser.parse_args()

if len(args) != 2:
    parser.print_help()
    sys.exit(1)

eventsInfo = args[0]
decodedEventInfos = decodeEventsInfo( eventsInfo )

(dbsQuery,eventRanges) = buildDBSQuery( decodedEventInfos )

cfg = args[1]

handle = open(cfg, 'r')
cfo = imp.load_source("pycfg", cfg, handle)
process = cfo.process
handle.close()

print dbsQuery
dbsOut = os.popen('dbs search --query %s' % dbsQuery)

process.source.fileNames = getFiles( dbsOut )


process.source.eventsToProcess = eventRanges

outFile = open("tmpConfig.py","w")
outFile.write("import FWCore.ParameterSet.Config as cms\n")
outFile.write(process.dumpPython())
outFile.close()

if options.negate == False:
    os.system("cmsRun tmpConfig.py")
