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

######################################################
# functions definitions


#########################
# get good B field runs from RunInfo DB
def getGoodBRuns(options):

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
            if v>0 or x[0]==67647L or x[0]==66893L or x[0]==67264L:
                print x[0],x[1] ,x[2], x[2][4], x[2][3]
                #print x[0],x[1] ,x[2], x[2][4], timeStamptoUTC(x[2][6]), timeStamptoUTC(x[2][7])
            if x[2][4] >= minI and x[2][3] <= maxI:
                runs_b_on.append(int(x[0]))

    except Exception, er :
        print er

    print "### runs with good B field ###"
    print runs_b_on

    return runs_b_on


#########################
# obtaining list of good quality runs

def getGoodQRuns(options):

    runs_good_dq = []

    dbs_quiery = "find run where dataset="+options.dqDataset+" and dq="+options.dqCriteria

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
                   default=0L,
                   dest="startRun")

parser.add_option("-e", "--endRun",
                   help="Last run number in range.",
                   type="int",
                   default=999999999L,
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
                   "'DT_Shift_Offline=GOOD&CSC_Shift_Offline=GOOD&SiStrip_Shift_Offline=GOOD&Pixel_Shift_Offline=GOOD'",
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


copyargs = sys.argv[:]
for i in range(len(copyargs)):
    if copyargs[i] == "":
        copyargs[i] = "\"\""

infotofile = ["### %s\n" % " ".join(copyargs)]

allOptions = '### ' + copyargs[0] + ' --alcaDataset ' + options.alcaDataset + ' --isMC ' + options.isMC + \
             ' --startRun ' + str(options.startRun) + ' --endRun '+ str(options.endRun) + \
             ' --minB ' + str(options.minB) + ' --maxB ' + str(options.maxB) + \
             ' --dbTag ' + options.dbTag + ' --dqDataset ' + options.dqDataset + ' --dqCriteria "' + options.dqCriteria + '"'\
             ' --outputFile ' + options.outputFile

print "### all options, including default:"
print allOptions



######################################################
# get good B field runs from RunInfo DB

runs_b_on = []

if options.isMC=='false':
    runs_b_on = getGoodBRuns(options)

    infotofile.append("### runs with good B field ###\n")
    infotofile.append("### %s\n" % str(runs_b_on))

######################################################
# Add requiremment of good quality runs

runs_good_dq = []
runs_good = []

if options.isMC=='false':
    runs_good_dq = getGoodQRuns(options)
        
    infotofile.append("### runs with good quality ###\n")
    infotofile.append("### %s\n" % str(runs_good_dq))

    # find intersection of runs_b_on and runs_good_dq
    runs_good = [val for val in runs_b_on if val in runs_good_dq]

    print "### runs with good B field and quality ###"
    print runs_good

    infotofile.append("### runs with good B field and quality ###\n")
    infotofile.append("### %s\n" % str(runs_good))

######################################################
# Find files for good runs

dbs_quiery = "find run, file.numevents, file where dataset="+options.alcaDataset+" and run>="+str(options.startRun)+" and run <="+str(options.endRun)

os.system('python $DBSCMD_HOME/dbsCommandLine.py -c  search --noheader --query="'+dbs_quiery+'" | sort > /tmp/runs_and_files_full_of_pink_bunnies')

list_of_files = []
list_of_runs = []
total_numevents = 0

ff = open('/tmp/runs_and_files_full_of_pink_bunnies','r')
for line in ff:
    (run, numevents, fname) = line.split('   ')
    if options.isMC=='false' and (int(run) not in runs_good):
        continue
    fname = fname.rstrip('\n')
    list_of_files.append(fname)
    list_of_runs.append(run)
    total_numevents += int(numevents)
ff.close()
os.system('rm /tmp/runs_and_files_full_of_pink_bunnies')
print "### total number of events in those runs = "+str(total_numevents)

infotofile.append("### total number of events in those runs = "+str(total_numevents))

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
    ff.write("    '"+ list_of_files[i] +"'"+comma+" # "+ list_of_runs[i] +"\n")
ff.write(']\n')
ff.close()




