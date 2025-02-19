#!/usr/bin/env python

import sys
import os

#Check arg,settings

if len(sys.argv) != 2 : 
    print """
    Usage: create_harvesting_py.py  <dataset>
    example:
    create_harvesting_py.py \
     /RelValTTbar/CMSSW_3_1_0_pre4_STARTUP_30X_v1/GEN-SIM-RECO
    """
    sys.exit(10) 
    
#Get data files of dataset to be processed
if os.getenv('DBSCMD_HOME','NOTSET') == 'NOTSET' :
    print "dbs not set!"
    sys.exit(11)

if os.getenv('CMSSW_VERSION','NOTSET') == 'NOTSET' :
    print """
    cmssw not set!
    example:
      cmsrel CMSSW_3_1_0_pre4
      cd CMSSW_3_1_0_pre4/src
      eval `scramv1 runtime -sh`
      cd -
    """
    sys.exit(12) 

dsetpath = sys.argv[1]

from DBSAPI.dbsApi import DbsApi
from DBSAPI.dbsException import *
from DBSAPI.dbsApiException import *
from DBSAPI.dbsOptions import DbsOptionParser

optManager  = DbsOptionParser()
(opts,args) = optManager.getOpt()
api = DbsApi(opts.__dict__)

print "dataset: ", dsetpath
print "data files: "
for afile in api.listFiles(path=dsetpath):
  print "  %s" % afile['LogicalFileName']

#Determine number of events/processes
totnevts=0
for afile in api.listFiles(path=dsetpath):
  totnevts += afile['NumberOfEvents']
njobs = 1
nevtref = 9000
if totnevts > nevtref : njobs = (int) (totnevts / 9000)
print "Total # events: ", totnevts, \
      " to be executed in ", njobs, "processes"


#Run cmsDriver command
raw_cmsdriver = "cmsDriver.py harvest -s HARVESTING:validationHarvesting --mc  --conditions FrontierConditions_GlobalTag,STARTUP_30X::All --harvesting AtJobEnd --no_exec -n -1"

print "executing cmsdriver command:\n\t", raw_cmsdriver

os.system( '`' + raw_cmsdriver + '`' )


#Open output py
fin_name="harvest_HARVESTING_STARTUP.py"
pyout_name = "harvest.py"
os.system("touch " + fin_name)
os.system('mv ' + fin_name + " " + pyout_name )
pyout = open(pyout_name, 'a')

#Added to py config: input, output file name, dqm settings
pyout.write("\n\n##additions to cmsDriver output \n")
pyout.write("#DQMStore.referenceFileName = ''\n")
pyout.write("process.dqmSaver.workflow = '" + dsetpath + "'\n")
pyout.write("process.source.fileNames = cms.untracked.vstring(\n")

for afile in api.listFiles(path=dsetpath):
    pyout.write("  '%s',\n" % afile['LogicalFileName'])

pyout.write(")")
pyout.close()


#Create crab conf

crab_block = """
[CRAB]
jobtype = cmssw
scheduler = glite
#server_name = 

[EDG]
remove_default_blacklist=1
rb = CERN

[USER]
return_data = 0
copy_data = 1
storage_element=srm-cms.cern.ch
storage_path=/srm/managerv2?SFN=/castor/cern.ch/
user_remote_dir=/user/n/nuno/relval/harvest/
publish_data=0
thresholdLevel=70
eMail=nuno@cern.ch

[CMSSW]
total_number_of_events=-1
show_prod = 1
"""

crab_name="crab.cfg"
os.system("touch " + crab_name)
os.system("mv " + crab_name + " " + crab_name + "_old")

crab_cfg = open(crab_name, 'w')
crab_cfg.write(crab_block)

rootfile = "DQM_V0001_R000000001" \
           + dsetpath.replace('/','__') \
           + ".root"

crab_cfg.write("number_of_jobs=" + str(njobs) + "\n")
crab_cfg.write("pset=" + pyout_name + "\n")
crab_cfg.write("output_file=" + rootfile + "\n")
crab_cfg.write("datasetpath=" + dsetpath + "\n")


crab_cfg.close()

#os.system("cat " + pyout_name)
#print "Created crab conf:\t", crab_name,"\n"

print '\n\nCreated:\n\t %(pwd)s/%(pf)s \n\t %(pwd)s/%(cf)s' \
      % {'pwd' : os.environ["PWD"], 'pf' : pyout_name, 'cf' : crab_name}

print "Done."
