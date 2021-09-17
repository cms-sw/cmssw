#!/usr/bin/env python3

# Anzar Afaq         June 17, 2008
# Oleksiy Atramentov June 21, 2008
# Charles Plager     Sept  7, 2010
# Volker Adler       Apr  16, 2014
# Raman Khurana      June 18, 2015
# Dinko Ferencek     June 27, 2015
from __future__ import print_function
import os
import sys
import optparse
import re
import commands
from FWCore.PythonUtilities.LumiList   import LumiList
import json
from pprint import pprint
from datetime import datetime
import subprocess
import Utilities.General.cmssw_das_client as das_client
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
            raise RuntimeError("Can not parse '%s' as Event object" \
                  % line.strip())
        if not self['dataset']:
            print("No dataset is defined for '%s'.  Aborting." % line.strip())
            raise RuntimeError('Missing dataset')

    def __getattr__ (self, key):
        return self[key]

    def __str__ (self):
        return "run = %(run)i, lumi = %(lumi)i, event = %(event)i, dataset = %(dataset)s"  % self


#################
## Subroutines ##
#################

def getFileNames(event, client=None):
    """Return files for given DAS query"""
    if  client == 'das_client':
        return getFileNames_das_client(event)
    elif client == 'dasgoclient':
        return getFileNames_dasgoclient(event)
    # default action
    for path in os.getenv('PATH').split(':'):
        if  os.path.isfile(os.path.join(path, 'dasgoclient')):
            return getFileNames_dasgoclient(event)
    return getFileNames_das_client(event)

def getFileNames_das_client(event):
    """Return files for given DAS query via das_client"""
    files = []

    query = "file dataset=%(dataset)s run=%(run)i lumi=%(lumi)i | grep file.name" % event
    jsondict = das_client.get_data(query)
    status = jsondict['status']
    if status != 'ok':
        print("DAS query status: %s"%(status))
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

def getFileNames_dasgoclient(event):
    """Return files for given DAS query via dasgoclient"""
    query = "file dataset=%(dataset)s run=%(run)i lumi=%(lumi)i" % event
    cmd = ['dasgoclient', '-query', query, '-json']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    files = []
    err = proc.stderr.read()
    if  err:
        print("DAS error: %s" % err)
    else:
        for row in json.load(proc.stdout):
            for rec in row.get('file', []):
                fname = rec.get('name', '')
                if fname:
                    files.append(fname)
    return files

def fullCPMpath():
    base = os.environ.get ('CMSSW_BASE')
    if not base:
        raise RuntimeError("CMSSW Environment not set")
    retval = "%s/src/PhysicsTools/Utilities/configuration/copyPickMerge_cfg.py" \
             % base
    if os.path.exists (retval):
        return retval
    base = os.environ.get ('CMSSW_RELEASE_BASE')
    retval = "%s/src/PhysicsTools/Utilities/configuration/copyPickMerge_cfg.py" \
             % base
    if os.path.exists (retval):
        return retval
    raise RuntimeError("Could not find copyPickMerge_cfg.py")

def guessEmail():
    return '%s@%s' % (commands.getoutput ('whoami'),
                      '.'.join(commands.getoutput('hostname').split('.')[-2:]))

def setupCrabDict (options):
    date = datetime.now().strftime('%Y%m%d_%H%M%S')
    crab = {}
    base = options.base
    crab['runEvent']        = '%s_runEvents.txt' % base
    crab['copyPickMerge']   = fullCPMpath()
    crab['output']          = '%s.root' % base
    crab['crabcfg']         = '%s_crab.py' % base
    crab['json']            = '%s.json' % base
    crab['dataset']         = Event.dataset
    crab['email']           = options.email
    crab['WorkArea']        = date
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
## Edited By Raman Khurana
##
## CRAB documentation : https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideCrab
##
## CRAB 3 parameters : https://twiki.cern.ch/twiki/bin/view/CMSPublic/CRAB3ConfigurationFile#CRAB_configuration_parameters
##
## Once you are happy with this file, please run
## crab submit

## In CRAB3 the configuration file is in Python language. It consists of creating a Configuration object imported from the WMCore library: 

from WMCore.Configuration import Configuration
config = Configuration()

##  Once the Configuration object is created, it is possible to add new sections into it with corresponding parameters
config.section_("General")
config.General.requestName = 'pickEvents'
config.General.workArea = 'crab_pickevents_%(WorkArea)s'


config.section_("JobType")
config.JobType.pluginName = 'Analysis'
config.JobType.psetName = '%(copyPickMerge)s'
config.JobType.pyCfgParams = ['eventsToProcess_load=%(runEvent)s', 'outputFile=%(output)s']

config.section_("Data")
config.Data.inputDataset = '%(dataset)s'

config.Data.inputDBS = 'global'
config.Data.splitting = 'LumiBased'
config.Data.unitsPerJob = 5
config.Data.lumiMask = '%(json)s'
#config.Data.publication = True
#config.Data.publishDbsUrl = 'phys03'
#config.Data.publishDataName = 'CRAB3_CSA_DYJets'
#config.JobType.allowNonProductionCMSSW=True

config.section_("Site")
## Change site name accordingly
config.Site.storageSite = "T2_US_Wisconsin"

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
https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookPickEvents ''')
    parser.add_option ('--output', dest='base', type='string',
                       default='pickevents',
                       help='Base name to use for output files (root, JSON, run and event list, etc.; default "%default")')
    parser.add_option ('--runInteractive', dest='runInteractive', action='store_true',
                       help = 'Call "cmsRun" command if possible.  Can take a long time.')
    parser.add_option ('--printInteractive', dest='printInteractive', action='store_true',
                       help = 'Print "cmsRun" command instead of running it.')
    parser.add_option ('--maxEventsInteractive', dest='maxEventsInteractive', type='int',
                       default=20,
                       help = 'Maximum number of events allowed to be processed interactively.')
    parser.add_option ('--crab', dest='crab', action='store_true',
                       help = 'Force CRAB setup instead of interactive mode')
    parser.add_option ('--crabCondor', dest='crabCondor', action='store_true',
                       help = 'Tell CRAB to use Condor scheduler (FNAL or OSG sites).')
    parser.add_option ('--email', dest='email', type='string',
                       default='',
                       help="Specify email for CRAB (default '%s')" % email )
    das_cli = ''
    parser.add_option ('--das-client', dest='das_cli', type='string',
                       default=das_cli,
                       help="Specify das client to use (default '%s')" % das_cli )
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
                raise RuntimeError("'%s' is not a proper event" % piece)
            eventList.append (event)
    else:
        # read events from file
        source = open(args[0], 'r')
        for line in source:
            line = commentRE.sub ('', line)
            try:
                event = Event (line)
            except:
                print("Skipping '%s'." % line.strip())
                continue
            eventList.append(event)
        source.close()

    if not eventList:
        print("No events defined.  Aborting.")
        sys.exit()

    if len (eventList) > options.maxEventsInteractive:
        options.crab = True

    if options.crab:

        ##########
        ## CRAB ##
        ##########
        if options.runInteractive:
            raise RuntimeError("This job cannot be run interactively, but rather by crab.  Please call without the '--runInteractive' flag or increase the '--maxEventsInteractive' value.")
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
        print("Please visit CRAB twiki for instructions on how to setup environment for CRAB:\nhttps://twiki.cern.ch/twiki/bin/viewauth/CMS/SWGuideCrab\n")
        if options.crabCondor:
            print("You are running on condor.  Please make sure you have read instructions on\nhttps://twiki.cern.ch/twiki/bin/view/CMS/CRABonLPCCAF\n")
            if not os.path.exists ('%s/.profile' % os.environ.get('HOME')):
                print("** WARNING: ** You are missing ~/.profile file.  Please see CRABonLPCCAF instructions above.\n")
        print("Setup your environment for CRAB and edit %(crabcfg)s to make any desired changed.  Then run:\n\ncrab submit -c %(crabcfg)s\n" % crabDict)

    else:

        #################
        ## Interactive ##
        #################
        files = []
        eventPurgeList = []
        for event in eventList:
            eventFiles = getFileNames(event, options.das_cli)
            if eventFiles == ['[]']: # event not contained in the input dataset
                print("** WARNING: ** According to a DAS query, run = %i; lumi = %i; event = %i not contained in %s.  Skipping."%(event.run,event.lumi,event.event,event.dataset))
                eventPurgeList.append( event )
            else:
                files.extend( eventFiles )
        # Purge events
        for event in eventPurgeList:
            eventList.remove( event )
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
        print("\n%s" % command)
        if options.runInteractive and not options.printInteractive:
            os.system (command)

