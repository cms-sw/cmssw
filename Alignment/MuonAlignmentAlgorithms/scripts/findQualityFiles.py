#! /usr/bin/env python


import os,sys, DLFCN
import optparse

copyargs = sys.argv[:]
for i in range(len(copyargs)):
    if copyargs[i] == "":
        copyargs[i] = "\"\""

######################################################
# To parse commandline args

usage='%prog [options]\n\n'+\
    'Creates a Python configuration file with filenames for runs in specified run range, with certain min B field and data quality requirements.'

parser=optparse.OptionParser(usage)

parser.add_option("-s", "--startRun",
                   help="First run number in range.",
                   type="int",
                   default=1L,
                   dest="startRun")

parser.add_option("-e", "--endRun",
                   help="Last run number in range.",
                   type="int",
                   default=999999999L,
                   dest="endRun")

parser.add_option("-b", "--minB",
                   help="Lower limit on minimal B field for a run.",
                   type="float",
                   default=3.77,
                   dest="minB")

parser.add_option("--maxB",
                   help="Upper limit on B field for a run.",
                   type="float",
                   default=999.,
                   dest="maxB")

parser.add_option("-t", "--dbTag",
                   help="DB tag to use.",
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
                   help="Name of the dataset for choosing good data quality runs.\n"+\
                   "For CRAFT08 use '/Cosmics/Commissioning08-v1/RAW'\n"+\
                   "For CRAFT09 use '/Cosmics/CRAFT09-v1/RAW'",
                   type="string",
                   #default="/Cosmics/Commissioning08-v1/RAW",
                   default="/Cosmics/CRAFT09-v1/RAW",
                   dest="dqDataset")

parser.add_option("--dqCriteria",
                   help="Set of DQ criteria to use with -dq flag of dbs.\n"+\
                   "An example of a really strict condition:\n"
                   "'DT_Global=GOOD&CSC_Global=GOOD&SiStrip_Global=GOOD&Pixel_Global=GOOD'",
                   type="string",
                   #default="DT_Global=GOOD&SiStrip_Global=GOOD&Pixel_Global=GOOD",
                   #default="DT_Global=GOOD&Pixel_Global=GOOD",
                   default="DT_Global=GOOD",
                   dest="dqCriteria")

parser.add_option("--alcaDataset",
                   help="Name of the input AlCa dataset to get filenames from.",
                   type="string",
                   #default="/Cosmics/Commissioning08-2213_Tosca090322_2pi_scaled_ReReco_FromTrackerPointing-v1/RAW-RECO",
                   default="/Cosmics/Commissioning08_CRAFT_ALL_V11_StreamALCARECOMuAlGlobalCosmics_227_Tosca090216_ReReco_FromTrackerPointing_v5/ALCARECO",
                   dest="alcaDataset")

parser.add_option("--outputFile",
                   help="Name for output file (please include the .py suffix)",
                   type="string",
                   default="filelist",
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

v = options.verbose

minI = options.minB*18160/3.8
maxI = options.maxB*18160/3.8

infotofile = ["### %s\n" % " ".join(copyargs)]

######################################################
# RunInfo DB connection setup

sys.setdlopenflags(DLFCN.RTLD_GLOBAL+DLFCN.RTLD_LAZY)

from pluginCondDBPyInterface import *
a = FWIncantation()
#os.putenv("CORAL_AUTH_PATH","/afs/cern.ch/cms/DB/conddb")
rdbms = RDBMS("/afs/cern.ch/cms/DB/conddb")

from CondCore.Utilities import iovInspector as inspect
from CondCore.Utilities.timeUnitHelper import *

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

runs_b_on = []

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

infotofile.append("### runs with good B field ###\n")
infotofile.append("### %s\n" % str(runs_b_on))

######################################################
# Add requiremment of good quality runs

runs_good_dq = []

dbs_quiery = "find run where dataset="+options.dqDataset+" and dq="+options.dqCriteria

os.system('python $DBSCMD_HOME/dbsCommandLine.py -c  search --noheader --query="'+dbs_quiery+'" | sort > /tmp/runs_full_of_pink_bunnies')

ff = open('/tmp/runs_full_of_pink_bunnies', "r")
line = ff.readline()
while line and line!='':
    runs_good_dq.append(int(line))
    line = ff.readline()
ff.close()

os.system('rm /tmp/runs_full_of_pink_bunnies')

print "### runs with good quality ###"
print runs_good_dq

infotofile.append("### runs with good quality ###\n")
infotofile.append("### %s\n" % str(runs_good_dq))

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
    if int(run) not in runs_good:
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

