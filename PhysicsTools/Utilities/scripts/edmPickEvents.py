#!/usr/bin/env python

# Anzar Afaq         June 17, 2008
# Oleksiy Atramentov June 21, 2008
# Charles Plager     Sept  7, 2010

import os
import sys
import optparse
import re
import commands
import xml.sax
import xml.sax.handler
from FWCore.PythonUtilities.LumiList   import LumiList
from xml.sax import SAXParseException
from DBSAPI.dbsException import *
from DBSAPI.dbsApiException import *
from DBSAPI.dbsOptions import DbsOptionParser
from DBSAPI.dbsApi import DbsApi
from pprint import pprint


help = """
How to use: 

edmPickEvent.py event_list


event_list is a text file
'#' is valid comment character

run1 lumi_section1 event1
run2 lumi_section2 event2
...

For example:
46968   2      4
47011 105     23
47011 140  12312

run, lumi_section, and event are integers that you can get from
edm::Event(Auxiliary)

dataset: it just a name of the physics dataset, if you don't know exact name
    you can provide a mask, e.g.: *QCD*RAW (but it doesn't work at the moment)

For updated information see Wiki:
https://twiki.cern.ch/twiki/bin/view/CMS/PickEvents 
"""


########################
## Event helper class ##
########################

class Event (dict):

    defaultDataset = None

    def __init__ (self, line, **kwargs):
        pieces = line.strip().split()
        try:
            self['run']     = int( pieces[0] )
            self['lumi']    = int( pieces[1] )
            self['event']   = int( pieces[2] )
            self['dataset'] =  Event.defaultDataset
        except:
            raise RuntimeError, "Can not parse '%s' as Event object" \
                  % line.strip()
        if not self['dataset']:
            print "No dataset is defined for '%s'.  Aborting." % line.strip()
            raise RuntimeError, 'Missing dataset'

    def __getattr__ (self, key):
        return self[key]

    def __str__ (self):
        return "run = %(run)i, event = %(event)i, lumi = %(lumi)i, dataset = %(dataset)s"  % self
    

#################
## Subroutines ##
#################

def getFileNames (event, dbsOptions = {}):
    # Query DBS
    try:
        api = DbsApi (dbsOptions)
        query = "find file where dataset=%(dataset)s and run=%(run)i and lumi=%(lumi)i" % event

        xmldata = api.executeQuery(query)
    except DbsApiException, ex:
        print "Caught API Exception %s: %s "  % (ex.getClassName(), ex.getErrorMessage() )
        if ex.getErrorCode() not in (None, ""):
            print "DBS Exception Error Code: ", ex.getErrorCode()

    # Parse the resulting xml output.
    files = []
    try:
        class Handler (xml.sax.handler.ContentHandler):
            def __init__(self):
                self.inFile = 0
            def startElement(self, name, attrs):
                if name == 'file':
                    self.inFile = 1
            def endElement(self, name):
                if name == 'file':
                    self.inFile = 0
            def characters(self, data):
                if self.inFile:
                    files.append(str(data))

        ##  # I want to see what this looks like
        ##  print "xmldata", xmldata
        xml.sax.parseString (xmldata, Handler ())
    except SAXParseException, ex:
        msg = "Unable to parse XML response from DBS Server"
        msg += "\n  Server has not responded as desired, try setting level=DBSDEBUG"
        raise DbsBadXMLData(args=msg, code="5999")

    return files


def fullCPMpath():
    base = os.environ.get ('CMSSW_BASE')
    if not base:
        raise RuntimeError, "CMSSW Environment not set"
    retval = "%s/src/PhysicsTools/Utilities/configuration/copyPickMerge_cfg.py" \
             % base
    if os.path.exists (retval):
        return retval
    base = os.environ.get ('CMSSW_RELEASE_BASE')
    retval = "%s/src/PhysicsTools/Utilities/configuration/copyPickMerge_cfg.py" \
             % base
    if os.path.exists (retval):
        return retval
    raise RuntimeError, "Could not find copyPickMerge_cfg.py"


def setupCrabDict (options):
    crab = {}
    base = options.base
    crab['runEvent']      = '%s_runEvents.txt' % base
    crab['copyPickMerge'] = fullCPMpath()
    crab['output']        = '%s.root' % base
    crab['crabcfg']       = '%s_crab.config' % base
    crab['json']          = '%s.json' % base
    crab['dataset']       = options.dataset
    crab['email']         = '%s@%s' % (commands.getoutput('whoami'),
                                       commands.getoutput('hostname'))
    return crab


# crab template
crabTemplate = '''
# CRAB documentation:
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideCrab
#
# Once you are happy with this file, please run
# crab -create -cfg %(crabcfg)s
# crab -submit -cfg %(crabcfg)s

[CMSSW]
pycfg_params = eventsToProcess_load=%(runEvent)s outputFile=%(output)s

lumi_mask               = %(json)s
total_number_of_lumis   = -1
lumis_per_job           = 1
pset                    = %(copyPickMerge)s
datasetpath             = %(dataset)s
output_file             = %(output)s

[USER]
return_data             = 1
email                   = %(email)s

# if you want to copy the data or put it in a storage element, do it
# here.


[CRAB]
                          # use "condor" if you run on CAF at FNAL or USG
scheduler               = glite  
jobtype                 = cmssw
use_server              = 1
'''


########################
## ################## ##
## ## Main Program ## ##
## ################## ##
########################

if __name__ == "__main__":
    
    parser = optparse.OptionParser ("Usage: %prog [options] events.txt")
    parser.add_option ('--crab', dest='crab', action='store_true',
                       help = 'force CRAB setup instead of interactive mode')
    parser.add_option ('--base', dest='base', type='string',
                       default='pickevents',
                       help='Base of name to call output, JSON, etc. files')
    parser.add_option ('--dataset', dest='dataset', type='string',
                       default='', 
                       help='dataset to check')
    (options, args) = parser.parse_args()

    Event.defaultDataset = options.dataset
    
    if len(args) != 1:
        parser.print_help()
        sys.exit(0)

    eventList = []
    commentRE = re.compile (r'#.+$')
    source = open(args[0], 'r')
    for line in source:
        line = commentRE.sub ('', line)
        try:
            event = Event (line)
        except:
            print "Skipping '%s'." % line.strip()
            continue
        eventList.append(event)
    source.close()

    if len (eventList) > 20:
        options.crab = True

    if options.crab:

        ##########
        ## CRAB ##
        ##########
        runsAndLumis = [ (event.run, event.lumi) for event in eventList]
        json = LumiList (lumis = runsAndLumis)
        eventsToProcess = '\n'.join(\
          sorted( [ "%d:%d" % (event.run, event.event) for event in eventList ] ) )
        crabDict = setupCrabDict (options)
        json.writeJSON (crabDict['json'])
        target = open (crabDict['runEvent'], 'w')
        target.write ("%s\n" % eventsToProcess)
        target.close()
        target = open (crabDict['crabcfg'], 'w')
        target.write (crabTemplate % crabDict)
        target.close
        print "Edit %(crabcfg)s to make any desired changed.  The run:\ncrab -create -cfg %(crabcfg)s\ncrab -submit -cfg %(crabcfg)s\n" % crabDict

    else:

        #################
        ## Interactive ##
        #################    
        files = []
        for event in eventList:
            files.extend( getFileNames (event) )
        if not eventList:
            print "No events defind.  Aborting."
            sys.exit()
        # Purge duplicate files
        files = sorted( list( set( files ) ) )
        source = ','.join (files) + '\n'
        eventsToProcess = ','.join(\
          sorted( [ "%d:%d" % (event.run, event.event) for event in eventList ] ) )
        command = 'edmCopyPickMerge outputFile=%s.root \\\n  eventsToProcess=%s \\\n  inputFiles=%s' \
                  % (options.base, eventsToProcess, source)
        print "\n%s" % command
