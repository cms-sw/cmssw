#!/usr/bin/env python

# Anzar Afaq         June 17, 2008
# Oleksiy Atramentov June 21, 2008
# Charles Plager     Sept  7, 2010
# Volker Adler       Apr  16, 2014

import os
import sys
import optparse
import re
import commands
from FWCore.PythonUtilities.LumiList   import LumiList
import das_client
import json
from pprint import pprint


help = """
How to use:

edmPickEvent.py dataset run1:lumi1:event1 run2:lumi2:event2

- or -

edmPickEvent.py dataset listOfEvents.txt


listOfEvents is a text file:
# this line is ignored as a comment
# since '#' is a valid comment character
run1 lumi_section1 event1
run2 lumi_section2 event2

For example:
# run lum   event
46968   2      4
47011 105     23
47011 140  12312

run, lumi_section, and event are integers that you can get from
edm::Event(Auxiliary)

dataset: it just a name of the physics dataset, if you don't know exact name
    you can provide a mask, e.g.: *QCD*RAW

For updated information see Wiki:
https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookPickEvents
"""


########################
## Event helper class ##
########################

class Event (dict):

    dataset = None
    splitRE = re.compile (r'[\s:,]+')
    def __init__ (self, line, **kwargs):
        pieces = Event.splitRE.split (line.strip())
        try:
            self['run']     = int( pieces[0] )
            self['lumi']    = int( pieces[1] )
            self['event']   = int( pieces[2] )
            self['dataset'] =  Event.dataset
        except:
            raise RuntimeError, "Can not parse '%s' as Event object" \
                  % line.strip()
        if not self['dataset']:
            print "No dataset is defined for '%s'.  Aborting." % line.strip()
            raise RuntimeError, 'Missing dataset'

    def __getattr__ (self, key):
        return self[key]

    def __str__ (self):
        return "run = %(run)i, lumi = %(lumi)i, event = %(event)i, dataset = %(dataset)s"  % self


#################
## Subroutines ##
#################

def getFileNames (event):
    files = []
    # Query DAS
    query = "file dataset=%(dataset)s run=%(run)i lumi=%(lumi)i | grep file.name" % event
    jsondict = das_client.get_data('https://cmsweb.cern.ch', query, 0, 0, False)
    status = jsondict['status']
    if status != 'ok':
        print "DAS query status: %s"%(status)
        return files

    mongo_query = jsondict['mongo_query']
    filters = mongo_query['filters']
    data = jsondict['data']

    files = []
    for row in data:
        file = [r for r in das_client.get_value(row, filters['grep'])][0]
        if len(file) > 0 and not file in files:
            files.append(file)

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

def guessEmail():
    return '%s@%s' % (commands.getoutput ('whoami'),
                      '.'.join(commands.getoutput('hostname').split('.')[-2:]))


def setupCrabDict (options):
    crab = {}
    base = options.base
    crab['runEvent']      = '%s_runEvents.txt' % base
    crab['copyPickMerge'] = fullCPMpath()
    crab['output']        = '%s.root' % base
    crab['crabcfg']       = '%s_crab.config' % base
    crab['json']          = '%s.json' % base
    crab['dataset']       = Event.dataset
    crab['email']         = options.email
    if options.crabCondor:
        crab['scheduler'] = 'condor'
#        crab['useServer'] = ''
    else:
        crab['scheduler'] = 'remoteGlidein'
#        crab['useServer'] = 'use_server              = 1'
    crab['useServer'] = ''
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
# use "glite" in general; you can "condor" if you run on CAF at FNAL or USG
# site AND you know the files are available locally
scheduler               = %(scheduler)s
jobtype                 = cmssw
%(useServer)s
'''


########################
## ################## ##
## ## Main Program ## ##
## ################## ##
########################

if __name__ == "__main__":
    email = guessEmail()
    parser = optparse.OptionParser ("Usage: %prog [options] dataset events_or_events.txt", description='''This program
facilitates picking specific events from a data set.  For full details, please visit
https://twiki.cern.ch/twiki/bin/view/CMS/PickEvents ''')
    parser.add_option ('--output', dest='base', type='string',
                       default='pickevents',
                       help='Base name to use for output files (root, JSON, run and event list, etc.; default "%default")')
    parser.add_option ('--runInteractive', dest='runInteractive', action='store_true',
                       help = 'Call "cmsRun" command if possible.  Can take a long time.')
    parser.add_option ('--printInteractive', dest='printInteractive', action='store_true',
                       help = 'Print "cmsRun" command instead of running it.')
    parser.add_option ('--crab', dest='crab', action='store_true',
                       help = 'Force CRAB setup instead of interactive mode')
    parser.add_option ('--crabCondor', dest='crabCondor', action='store_true',
                       help = 'Tell CRAB to use Condor scheduler (FNAL or OSG sites).')
    parser.add_option ('--email', dest='email', type='string',
                       default='',
                       help="Specify email for CRAB (default '%s')" % email )
    (options, args) = parser.parse_args()


    if len(args) < 2:
        parser.print_help()
        sys.exit(0)

    if not options.email:
        options.email = email

    Event.dataset = args.pop(0)
    commentRE = re.compile (r'#.+$')
    colonRE   = re.compile (r':')
    eventList = []
    if len (args) > 1 or colonRE.search (args[0]):
        # events are coming in from the command line
        for piece in args:
            try:
                event = Event (piece)
            except:
                raise RuntimeError, "'%s' is not a proper event" % piece
            eventList.append (event)
    else:
        # read events from file
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
        if options.runInteractive:
            raise RuntimeError, "This job is can not be run interactive, but rather by crab.  Please call without '--runInteractive' flag."
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
        print "Please visit CRAB twiki for instructions on how to setup environment for CRAB:\nhttps://twiki.cern.ch/twiki/bin/viewauth/CMS/SWGuideCrab\n"
        if options.crabCondor:
            print "You are running on condor.  Please make sure you have read instructions on\nhttps://twiki.cern.ch/twiki/bin/view/CMS/CRABonLPCCAF\n"
            if not os.path.exists ('%s/.profile' % os.environ.get('HOME')):
                print "** WARNING: ** You are missing ~/.profile file.  Please see CRABonLPCCAF instructions above.\n"
        print "Setup your environment for CRAB.  Then edit %(crabcfg)s to make any desired changed.  The run:\n\ncrab -create -cfg %(crabcfg)s\ncrab -submit\n" % crabDict

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
        fileSet = set()
        uniqueFiles = []
        for filename in files:
            if filename in fileSet:
                continue
            fileSet.add (filename)
            uniqueFiles.append (filename)
        source = ','.join (uniqueFiles) + '\n'
        eventsToProcess = ','.join(\
          sorted( [ "%d:%d" % (event.run, event.event) for event in eventList ] ) )
        command = 'edmCopyPickMerge outputFile=%s.root \\\n  eventsToProcess=%s \\\n  inputFiles=%s' \
                  % (options.base, eventsToProcess, source)
        print "\n%s" % command
        if options.runInteractive and not options.printInteractive:
            os.system (command)

