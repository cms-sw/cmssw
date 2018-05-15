#! /usr/bin/env python

######################################################
### See documentation at
### https://twiki.cern.ch/twiki/bin/view/CMS/FindQualityFilesPy
### also run it with -h option
######################################################

import os,sys, DLFCN
import optparse

# for RunInfo API
from pluginCondDBPyInterface import *
from CondCore.Utilities import iovInspector as inspect
from CondCore.Utilities.timeUnitHelper import *

# for RunRegistry API
import xmlrpclib

# for json support
try: # FUTURE: Python 2.6, prior to 2.6 requires simplejson
    import json
except:
    try:
        import simplejson as json
    except:
        print "Please use lxplus or set an environment (for example crab) with json lib available"
        sys.exit(1)

######################################################
print "### command line:"
copyargs = sys.argv[:]
for i in range(len(copyargs)):
  if copyargs[i] == "":
    copyargs[i] = "\"\""
  if copyargs[i].find(" ") != -1:
    copyargs[i] = "\"%s\"" % copyargs[i]
commandline = " ".join(copyargs)

print commandline
infotofile = ["### %s\n" % commandline]

######################################################
# To parse commandline args

usage='%prog [options]\n\n'+\
    'Creates a Python configuration file with filenames for runs in specified run range, with certain min B field and data quality requirements.'

parser=optparse.OptionParser(usage)

parser.add_option("-d", "--alcaDataset",
                   help="[REQUIRED] Name of the input AlCa dataset to get filenames from.",
                   type="string",
                   #default="/Cosmics/Commissioning08-2213_Tosca090322_2pi_scaled_ReReco_FromTrackerPointing-v1/RAW-RECO",
                   #default="/Cosmics/Commissioning08_CRAFT_ALL_V11_StreamALCARECOMuAlGlobalCosmics_227_Tosca090216_ReReco_FromTrackerPointing_v5/ALCARECO",
                   default='',
                   dest="alcaDataset")

parser.add_option("-m", "--isMC",
                   help="Whether sample is MC (true) or real data (false).",
                   type="string",
                   default="false",
                   dest="isMC")

parser.add_option("-s", "--startRun",
                   help="First run number in range.",
                   type="int",
                   default=0,
                   dest="startRun")

parser.add_option("-e", "--endRun",
                   help="Last run number in range.",
                   type="int",
                   default=999999999,
                   dest="endRun")

parser.add_option("-b", "--minB",
                   help="Lower limit on minimal B field for a run.",
                   type="float",
                   #default=3.77,
                   default=0.,
                   dest="minB")

parser.add_option("--maxB",
                   help="Upper limit on B field for a run.",
                   type="float",
                   default=999.,
                   dest="maxB")

parser.add_option("-r","--runRegistry",
                   help="If present, use RunRegistry API for B field and data quality quiery",
                   action="store_true",
                   default=False,
                   dest="runRegistry")

parser.add_option("-j","--json",
                   help="If present with JSON file as argument, use JSON file for the good runs and ignore B field and --runRegistry options. "+\
                   "The latest JSON file is available at /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions10/7TeV/StreamExpress/",
                   type="string",
                   default="",
                   dest="json")

parser.add_option("-t", "--dbTag",
                   help="Runinfo DB tag to use.",
                   type="string",
                   default="runinfo_31X_hlt",
                   dest="dbTag")

parser.add_option("--printTags",
                   help="If present, the only thing script will do is printing list of tags in the DB",
                   action="store_true",
                   default=False,
                   dest="printTags")

parser.add_option("--dbName",
                   help="RunInfo DB name to use. The default one is "+\
                   "'oracle://cms_orcoff_prod/CMS_COND_31X_RUN_INFO'",
                   type="string",
                   default="oracle://cms_orcoff_prod/CMS_COND_31X_RUN_INFO",
                   dest="dbName")

parser.add_option("--dqDataset",
                   help="Dataset name to query for good data quality runs. "+\
                   "If this option is not used, dqDataset=alcaDataset is automatically set. "+\
                   "If alcaDataset does not have DQ information use /Cosmics/Commissioning08-v1/RAW for CRAFT08 "+\
                   "and use /Cosmics/CRAFT09-v1/RAW for CRAFT08",
                   type="string",
                   #default="/Cosmics/Commissioning08-v1/RAW",
                   #default="/Cosmics/CRAFT09-v1/RAW",
                   default="",
                   dest="dqDataset")

parser.add_option("-c", "--dqCriteria",
                   help="Set of DQ criteria to use with -dq flag of dbs.\n"+\
                   "An example of a really strict condition:\n"
                   "'DT_Shift_Offline=GOOD&CSC_Shift_Offline=GOOD&SiStrip_Shift_Offline=GOOD&Pixel_Shift_Offline=GOOD'"
                   "NOTE: if --runRegistry is used, DQ criteria sintax should be as Advanced query syntax for RR. E.g.:"
                   "\"{cmpDt}='GOOD' and {cmpCsc}='GOOD' and {cmpStrip}='GOOD' and {cmpPix}='GOOD'\"",
                   type="string",
                   #default="DT_Shift_Offline=GOOD&SiStrip_Shift_Offline=GOOD&Pixel_Shift_Offline=GOOD",
                   #default="DT_Shift_Offline=GOOD&Pixel_Shift_Offline=GOOD",
                   #default="DT_Shift_Offline=GOOD",
                   default="",
                   dest="dqCriteria")

parser.add_option("-o", "--outputFile",
                   help="Name for output file (please include the .py suffix)",
                   type="string",
                   default="filelist.py",
                   dest="outputFile")

parser.add_option("-v", "--verbose",
                   help="Degree of debug info verbosity",
                   type="int",
                   default=0,
                   dest="verbose")

options,args=parser.parse_args() 

#if '' in (options.infilename,
#          options.outfilename,
#          options.outputCommands):
#    raise ('Incomplete list of arguments!')


if options.alcaDataset=='' and not options.printTags:
    print "--alcaDataset /your/dataset/name is required!"
    sys.exit()
    
if options.dqDataset=='':
    options.dqDataset = options.alcaDataset

if not (options.isMC=='true' or options.isMC=='false'):
    print "--isMC option can have only 'true' or 'false' arguments"
    sys.exit()

v = options.verbose

minI = options.minB*18160/3.8
maxI = options.maxB*18160/3.8


rr = ''
if options.runRegistry: rr = ' --runRegistry'

jj = ''
if options.json!='': jj = ' --json '+options.json

allOptions = '### ' + copyargs[0] + ' --alcaDataset ' + options.alcaDataset + ' --isMC ' + options.isMC + \
             ' --startRun ' + str(options.startRun) + ' --endRun '+ str(options.endRun) + \
             ' --minB ' + str(options.minB) + ' --maxB ' + str(options.maxB) + rr + jj +\
             ' --dbTag ' + options.dbTag + ' --dqDataset ' + options.dqDataset + ' --dqCriteria "' + options.dqCriteria + '"'\
             ' --outputFile ' + options.outputFile

print "### all options, including default:"
print allOptions


######################################################
# functions definitions


#########################
# get good B field runs from RunInfo DB
def getGoodBRuns():

    runs_b_on = []

    sys.setdlopenflags(DLFCN.RTLD_GLOBAL+DLFCN.RTLD_LAZY)

    a = FWIncantation()
    #os.putenv("CORAL_AUTH_PATH","/afs/cern.ch/cms/DB/conddb")
    rdbms = RDBMS("/afs/cern.ch/cms/DB/conddb")

    db = rdbms.getDB(options.dbName)
    tags = db.allTags()

    if options.printTags:
        print  "\nOverview of all tags in "+options.dbName+" :\n"
        print tags
        print "\n"
        sys.exit()

    # for inspecting last run after run has started  
    #tag = 'runinfo_31X_hlt'
    tag = options.dbTag

    # for inspecting last run after run has stopped  
    #tag = 'runinfo_test'

    try :
        #log = db.lastLogEntry(tag)

        #for printing all log info present into log db 
        #print log.getState()

        iov = inspect.Iov(db,tag)
        #print "########overview of tag "+tag+"########"
        #print iov.list()
    
        if v>1 :
            print "######## summries ########"
            for x in  iov.summaries():
                print x[0], x[1], x[2] ,x[3]
    
        what={}
    
        if v>1 :
            print "###(start_current,stop_current,avg_current,max_current,min_current,run_interval_micros) vs runnumber###"
            print iov.trend(what)
    
        if v>0:
            print "######## trends ########"
        for x in iov.trendinrange(what,options.startRun-1,options.endRun+1):
            if v>0 or x[0]==67647 or x[0]==66893 or x[0]==67264:
                print x[0],x[1] ,x[2], x[2][4], x[2][3]
                #print x[0],x[1] ,x[2], x[2][4], timeStamptoUTC(x[2][6]), timeStamptoUTC(x[2][7])
            if x[2][4] >= minI and x[2][3] <= maxI:
                runs_b_on.append(int(x[0]))

    except Exception as er :
        print er

    print "### runs with good B field ###"
    print runs_b_on

    return runs_b_on


#########################
# obtaining list of good quality runs

def getGoodQRuns():

    runs_good_dq = []

    dbs_quiery = "find run where dataset="+options.dqDataset+" and dq="+options.dqCriteria
    print 'dbs search --noheader --query="'+dbs_quiery+'" | sort'

    os.system('python $DBSCMD_HOME/dbsCommandLine.py -c  search --noheader --query="'+dbs_quiery+'" | sort > /tmp/runs_full_of_pink_bunnies')

    #print 'python $DBSCMD_HOME/dbsCommandLine.py -c  search --noheader --query="'+dbs_quiery+'" | sort > /tmp/runs_full_of_pink_bunnies'

    ff = open('/tmp/runs_full_of_pink_bunnies', "r")
    line = ff.readline()
    while line and line!='':
        runs_good_dq.append(int(line))
        line = ff.readline()
    ff.close()

    os.system('rm /tmp/runs_full_of_pink_bunnies')

    print "### runs with good quality ###"
    print runs_good_dq

    return runs_good_dq

#########################
# obtaining list of good B and quality runs from Run Registry
# https://twiki.cern.ch/twiki/bin/view/CMS/DqmRrApi
# https://twiki.cern.ch/twiki/bin/viewauth/CMS/DQMRunRegistry

def getRunRegistryGoodRuns():

    server = xmlrpclib.ServerProxy('http://pccmsdqm04.cern.ch/runregistry/xmlrpc')
    
    rr_quiery = "{runNumber}>="+str(options.startRun)+" and {runNumber}<="+str(options.endRun)+\
                " and {bfield}>="+str(options.minB)+" and {bfield}<="+str(options.maxB)
    if options.dqCriteria != "": rr_quiery += " and "+options.dqCriteria
    
    rrstr = server.RunDatasetTable.export('GLOBAL', 'chart_runs_cum_evs_vs_bfield', rr_quiery)
    rrstr = rrstr.replace("bfield","'bfield'")
    rrstr = rrstr.replace("events","'events'")
    rrdata = eval(rrstr)

    runs_good = []
    for rr in rrdata['events']: runs_good.append(rr[0])

    return runs_good

#########################
# obtain a list of good runs from JSON file

def getJSONGoodRuns():

    # read json file
    jsonfile=file(options.json,'r')
    jsondict = json.load(jsonfile)

    runs_good = []
    for run in jsondict.keys(): runs_good.append(int(run))
    runs_good.sort()

    #mruns=[]
    #for run in jsondict.keys():
    #  if int(run)<144115 and int(run)>136034: mruns.append(int(run))
    #mruns.sort()
    #print len(mruns),"runs in \n",mruns
    
    return runs_good

######################################################
# get good B field runs from RunInfo DB

runs_b_on = []

if options.isMC=='false' and not options.runRegistry and options.json=='':
    runs_b_on = getGoodBRuns()

    infotofile.append("### runs with good B field ###\n")
    infotofile.append("### %s\n" % str(runs_b_on))

######################################################
# Add requiremment of good quality runs

runs_good_dq = []
runs_good = []

if options.isMC=='false' and not options.runRegistry and options.json=='':
    runs_good_dq = getGoodQRuns()
        
    infotofile.append("### runs with good quality ###\n")
    infotofile.append("### %s\n" % str(runs_good_dq))

    # find intersection of runs_b_on and runs_good_dq
    runs_good = [val for val in runs_b_on if val in runs_good_dq]

    print "### runs with good B field and quality ###"
    print runs_good

    infotofile.append("### runs with good B field and quality ###\n")
    infotofile.append("### %s\n" % str(runs_good))

######################################################
# use run registry API is specified

if options.isMC=='false' and options.runRegistry and options.json=='':
    runs_good = getRunRegistryGoodRuns()
    print "### runs with good B field and quality ###"
    print runs_good
    
    #infotofile.append("### runs with good B field and quality ###\n")
    #infotofile.append("### %s\n" % str(runs_good))

######################################################
# use JSON file if specified

if options.isMC=='false' and options.json!='':
    runs_good = getJSONGoodRuns()
    print "### good runs from JSON file ###"
    print runs_good

######################################################
# Find files for good runs

dbs_quiery = "find run, file.numevents, file where dataset="+options.alcaDataset+" and run>="+str(options.startRun)+" and run<="+str(options.endRun)+" and file.numevents>0"
#print 'dbs search --noheader --query="'+dbs_quiery+'" | sort'

os.system('python $DBSCMD_HOME/dbsCommandLine.py -c  search --noheader --query="'+dbs_quiery+'" | sort > /tmp/runs_and_files_full_of_pink_bunnies')

list_of_files = []
list_of_runs = []
list_of_numevents = []
total_numevents = 0

ff = open('/tmp/runs_and_files_full_of_pink_bunnies','r')
for line in ff:
    (run, numevents, fname) = line.split('   ')
    if options.isMC=='false' and (int(run) not in runs_good):
        continue
    fname = fname.rstrip('\n')
    list_of_files.append(fname)
    list_of_runs.append(int(run))
    list_of_numevents.append(numevents)
    total_numevents += int(numevents)
ff.close()
#os.system('rm /tmp/runs_and_files_full_of_pink_bunnies')

uniq_list_of_runs = sorted(set(list_of_runs))

print "### list of runs with good B field and quality in the dataset: ###"
print uniq_list_of_runs
infotofile.append("### list of runs with good B field and quality in the dataset: ###\n")
infotofile.append("### %s\n" % str(uniq_list_of_runs))


# prevent against duplication due to the fact now a file can have events from several runs
files_events = list(zip(list_of_files, list_of_numevents))
unique_files_events = list(set(files_events))
list_of_files, list_of_numevents = map(list, list(zip(*unique_files_events)))
total_numevents = sum( map(int, list_of_numevents) )

print "### total number of events in those "+str(len(uniq_list_of_runs))+" runs = "+str(total_numevents)

infotofile.append("### total number of events in those "+str(len(uniq_list_of_runs))+" runs = "+str(total_numevents))

######################################################
# Write out results

# ff = open(options.outputFile+'.txt','w')
size = len(list_of_files)
# for i in range(0,size):
#     ff.write(list_of_runs[i] + ", " + list_of_files[i]+"\n")
# ff.close()

ff = open(options.outputFile,'w')
ff.write("".join(infotofile))
ff.write("\nfileNames = [\n")
comma = ","
for i in range(0,size):
    if i==size-1:
        comma=""
    #ff.write("    '"+ list_of_files[i] +"'"+comma+" # "+ str(list_of_runs[i]) + "," + list_of_numevents[i] + "\n")
    ff.write("    '"+ list_of_files[i] +"'"+comma+" # "+ list_of_numevents[i] + "\n")
ff.write(']\n')
ff.close()

