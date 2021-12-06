#!/usr/bin/env python3
'''
Submits per run Primary Vertex Resoltion Alignment validation using the split vertex method,
usage:

submitPVResolutionJobs.py -i PVResolutionExample.ini -D /JetHT/Run2018C-TkAlMinBias-12Nov2019_UL2018-v2/ALCARECO
'''

from __future__ import print_function

__author__ = 'Marco Musich'
__copyright__ = 'Copyright 2020, CERN CMS'
__credits__ = ['Ernesto Migliore', 'Salvatore Di Guida']
__license__ = 'Unknown'
__maintainer__ = 'Marco Musich'
__email__ = 'marco.musich@cern.ch'
__version__ = 1

import os,sys
import getopt
import time
import json
import ROOT
import urllib
import string
import subprocess
import pprint
import warnings
from subprocess import Popen, PIPE
import multiprocessing
from optparse import OptionParser
import os, shlex, shutil, getpass
import configparser as ConfigParser

CopyRights  = '##################################\n'
CopyRights += '#    submitPVVResolutioJobs.py   #\n'
CopyRights += '#      marco.musich@cern.ch      #\n'
CopyRights += '#         October 2020           #\n'
CopyRights += '##################################\n'

##############################################
def get_status_output(*args, **kwargs):
##############################################
    p = subprocess.Popen(*args, **kwargs)
    stdout, stderr = p.communicate()
    return p.returncode, stdout, stderr

##############################################
def check_proxy():
##############################################
    """Check if GRID proxy has been initialized."""

    try:
        with open(os.devnull, "w") as dump:
            subprocess.check_call(["voms-proxy-info", "--exists"],
                                  stdout = dump, stderr = dump)
    except subprocess.CalledProcessError:
        return False
    return True

##############################################
def forward_proxy(rundir):
##############################################
    """Forward proxy to location visible from the batch system.
    Arguments:
    - `rundir`: directory for storing the forwarded proxy
    """

    if not check_proxy():
        print("Please create proxy via 'voms-proxy-init -voms cms -rfc'.")
        sys.exit(1)

    local_proxy = subprocess.check_output(["voms-proxy-info", "--path"]).strip()
    shutil.copyfile(local_proxy, os.path.join(rundir,".user_proxy"))

##############################################
def getFilesForRun(blob):
##############################################
    """
    returns the list of list files associated with a given dataset for a certain run
    """

    cmd2 = ' dasgoclient -limit=0 -query \'file run='+blob[0]+' dataset='+blob[1]+'\''
    q = Popen(cmd2 , shell=True, stdout=PIPE, stderr=PIPE)
    out, err = q.communicate()
    outputList = out.decode().split('\n')
    outputList.pop()
    return outputList 

##############################################
def write_HTCondor_submit_file(path, name, nruns, proxy_path=None):
##############################################
    """Writes 'job.submit' file in `path`.
    Arguments:
    - `path`: job directory
    - `script`: script to be executed
    - `proxy_path`: path to proxy (only used in case of requested proxy forward)
    """
        
    job_submit_template="""\
universe              = vanilla
executable            = {script:s}
output                = {jobm:s}/{out:s}.out
error                 = {jobm:s}/{out:s}.err
log                   = {jobm:s}/{out:s}.log
transfer_output_files = ""
+JobFlavour           = "{flavour:s}"
queue {njobs:s}
"""
    if proxy_path is not None:
        job_submit_template += """\
+x509userproxy        = "{proxy:s}"
"""
        
    job_submit_file = os.path.join(path, "job_"+name+".submit")
    with open(job_submit_file, "w") as f:
        f.write(job_submit_template.format(script = os.path.join(path,name+"_$(ProcId).sh"),
                                           out  = name+"_$(ProcId)",
                                           jobm = os.path.abspath(path),
                                           flavour = "tomorrow",
                                           njobs = str(nruns),
                                           proxy = proxy_path))

    return job_submit_file

##############################################
def getLuminosity(homedir,minRun,maxRun,isRunBased,verbose):
##############################################
    """Expects something like
    +-------+------+--------+--------+-------------------+------------------+
    | nfill | nrun | nls    | ncms   | totdelivered(/fb) | totrecorded(/fb) |
    +-------+------+--------+--------+-------------------+------------------+
    | 73    | 327  | 142418 | 138935 | 19.562            | 18.036           |
    +-------+------+--------+--------+-------------------+------------------+
    And extracts the total recorded luminosity (/b).
    """
    myCachedLumi={}
    if(not isRunBased):
        return myCachedLumi
    
    try:
        ## using normtag
        #output = subprocess.check_output([homedir+"/.local/bin/brilcalc", "lumi", "-b", "STABLE BEAMS", "--normtag","/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_PHYSICS.json", "-u", "/pb", "--begin", str(minRun),"--end",str(maxRun),"--output-style","csv"])

        ## no normtag
        output = subprocess.check_output([homedir+"/.local/bin/brilcalc", "lumi", "-b", "STABLE BEAMS","-u", "/pb", "--begin", str(minRun),"--end",str(maxRun),"--output-style","csv"])
    except:
        warnings.warn('ATTENTION! Impossible to query the BRIL DB!')
        return myCachedLumi

    if(verbose):
        print("INSIDE GET LUMINOSITY")
        print(output)

    for line in output.decode().split("\n"):
        if ("#" not in line):
            runToCache  = line.split(",")[0].split(":")[0] 
            lumiToCache = line.split(",")[-1].replace("\r", "")
            #print("run",runToCache)
            #print("lumi",lumiToCache)
            myCachedLumi[runToCache] = lumiToCache

    if(verbose):
        print(myCachedLumi)
    return myCachedLumi

##############################################
def isInJSON(run,jsonfile):
##############################################
    try:
        with open(jsonfile, 'r') as myJSON:
            jsonDATA = json.load(myJSON)
            return (run in jsonDATA)
    except:
        warnings.warn('ATTENTION! Impossible to find lumi mask! All runs will be used.')
        return True

#######################################################
def as_dict(config):
#######################################################
    dictionary = {}
    for section in config.sections():
        dictionary[section] = {}
        for option in config.options(section):
            dictionary[section][option] = config.get(section, option)

    return dictionary

#######################################################
def batchScriptCERN(theCMSSW_BASE,runindex, eosdir,lumiToRun,key,config):
#######################################################
    '''prepare the batch script, to run on HTCondor'''
    script = """#!/bin/bash
source /afs/cern.ch/cms/caf/setup.sh
CMSSW_DIR={CMSSW_BASE_DIR}/src/Alignment/OfflineValidation/test
echo "the mother directory is $CMSSW_DIR"
export X509_USER_PROXY=$CMSSW_DIR/.user_proxy
#OUT_DIR=$CMSSW_DIR/harvest ## for local storage
OUT_DIR={MYDIR}
LOG_DIR=$CMSSW_DIR/out
LXBATCH_DIR=`pwd`  
cd $CMSSW_DIR
eval `scram runtime -sh`
cd $LXBATCH_DIR 
cp -pr $CMSSW_DIR/cfg/PrimaryVertexResolution_{KEY}_{runindex}_cfg.py .
cmsRun PrimaryVertexResolution_{KEY}_{runindex}_cfg.py GlobalTag={GT} lumi={LUMITORUN} {REC} {EXT} >& log_{KEY}_run{runindex}.out
ls -lh . 
#for payloadOutput in $(ls *root ); do cp $payloadOutput $OUT_DIR/pvresolution_{KEY}_{runindex}.root ; done 
for payloadOutput in $(ls *root ); do xrdcp -f $payloadOutput root://eoscms/$OUT_DIR/pvresolution_{KEY}_{runindex}.root ; done
tar czf log_{KEY}_run{runindex}.tgz log_{KEY}_run{runindex}.out  
for logOutput in $(ls *tgz ); do cp $logOutput $LOG_DIR/ ; done 
""".format(CMSSW_BASE_DIR=theCMSSW_BASE,
           runindex=runindex,
           MYDIR=eosdir,
           KEY=key,
           LUMITORUN=lumiToRun,
           GT=config['globaltag'],
           EXT="external="+config['external'] if 'external' in config.keys() else "",
           REC="records="+config['records'] if 'records' in config.keys() else "")
   
    return script

#######################################################
# method to create recursively directories on EOS
#######################################################
def mkdir_eos(out_path):
    print("creating",out_path)
    newpath='/'
    for dir in out_path.split('/'):
        newpath=os.path.join(newpath,dir)
        # do not issue mkdir from very top of the tree
        if newpath.find('test_out') > 0:
            command="eos mkdir "+newpath
            p = subprocess.Popen(command,shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (out, err) = p.communicate()
            #print(out,err)
            p.wait()

    # now check that the directory exists
    command2="/afs/cern.ch/project/eos/installation/cms/bin/eos.select ls "+out_path
    p = subprocess.Popen(command2,shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (out, err) = p.communicate()
    p.wait()
    if p.returncode !=0:
        print(out)

##############################################
def main():
##############################################

    desc="""This is a description of %prog."""
    parser = OptionParser(description=desc,version='%prog version 0.1')
    parser.add_option('-s','--submit',  help='job submitted',       dest='submit',      action='store_true', default=False)
    parser.add_option('-j','--jobname', help='task name',           dest='taskname',    action='store',      default='myTask')
    parser.add_option('-i','--init',    help='ini file',            dest='iniPathName', action='store',      default="default.ini")
    parser.add_option('-b','--begin',   help='starting point',      dest='start',       action='store',      default='1')
    parser.add_option('-e','--end',     help='ending point',        dest='end',         action='store',      default='999999')
    parser.add_option('-D','--Dataset', help='dataset to run upon', dest='DATASET',     action='store',      default='/StreamExpressAlignment/Run2017F-TkAlMinBias-Express-v1/ALCARECO')
    parser.add_option('-v','--verbose', help='verbose output',      dest='verbose',     action='store_true', default=False)
    parser.add_option('-u','--unitTest',help='unit tests?',         dest='isUnitTest',  action='store_true', default=False)
    (opts, args) = parser.parse_args()

    global CopyRights
    print('\n'+CopyRights)

    input_CMSSW_BASE = os.environ.get('CMSSW_BASE')

    ## prepare the eos output directory

    USER = os.environ.get('USER')
    HOME = os.environ.get('HOME')
    eosdir=os.path.join("/store/group/alca_trackeralign",USER,"test_out",opts.taskname)
    if opts.submit:
        mkdir_eos(eosdir)
    else:
        print("Not going to create EOS folder. -s option has not been chosen")

    ## parse the configuration file

    try:
        config = ConfigParser.ConfigParser()
        config.read(opts.iniPathName)
    except ConfigParser.MissingSectionHeaderError as e:
        raise WrongIniFormatError(e)

    print("Parsed the following configuration \n\n")
    inputDict = as_dict(config)
    pprint.pprint(inputDict)

    if(not bool(inputDict)):
        raise SystemExit("\n\n ERROR! Could not parse any input file, perhaps you are submitting this from the wrong folder? \n\n")

    ## check first there is a valid grid proxy
    forward_proxy(".")

    #runs = commands.getstatusoutput("dasgoclient -query='run dataset="+opts.DATASET+"'")[1].split("\n")
    runs  = get_status_output("dasgoclient -query='run dataset="+opts.DATASET+"'",shell=True, stdout=PIPE, stderr=PIPE)[1].decode().split("\n")
    runs.pop()
    runs.sort()
    print("\n\n Will run on the following runs: \n",runs)

    if(not os.path.exists("cfg")):
        os.system("mkdir cfg")
        os.system("mkdir BASH")
        os.system("mkdir harvest")
        os.system("mkdir out")

    cwd = os.getcwd()
    bashdir = os.path.join(cwd,"BASH")

    runs.sort()

    ## check that the list of runs is not empty
    if(len(runs)==0):
        if(opts.isUnitTest):
            print('\n')
            print('=' * 70)
            print("|| WARNING: won't run on any run, probably DAS returned an empty query,\n|| but that's fine because this is a unit test!")
            print('=' * 70)
            print('\n')
            sys.exit(0)
        else:
            raise Exception('Will not run on any run.... please check again the configuration')
    else:
        # get from the DB the int luminosities
        myLumiDB = getLuminosity(HOME,runs[0],runs[-1],True,opts.verbose)

    if(opts.verbose):
        pprint.pprint(myLumiDB)

    lumimask = inputDict["Input"]["lumimask"]
    print("\n\n Using JSON file:",lumimask)

    mytuple=[]
    print("\n\n First run:",opts.start,"last run:",opts.end)

    for run in runs:
        if (int(run)<int(opts.start) or int(run)>int(opts.end)):
            print("excluding run",run)
            continue

        if not isInJSON(run,lumimask):
            continue

        else:
            print("'======> taking run",run)
            mytuple.append((run,opts.DATASET))

        #print mytuple

    pool = multiprocessing.Pool(processes=20)  # start 20 worker processes
    count = pool.map(getFilesForRun,mytuple)
    file_info = dict(zip(runs, count))

    if(opts.verbose):
        print(file_info)

    count=0
    for run in runs:
        count=count+1
        #if(count>10): 
        #    continue
        #run = run.strip("[").strip("]")

        if (int(run)<int(opts.start) or int(run)>int(opts.end)):
            print("excluding",run)
            continue

        if not isInJSON(run,lumimask):
            print("=====> excluding run:",run)
            continue

        files = file_info[run]
        if(opts.verbose):
            print(run, files)
        listOfFiles='['
        for ffile in files:
            listOfFiles=listOfFiles+"\""+str(ffile)+"\","
        listOfFiles+="]"
        
        #print(listOfFiles)

        theLumi='1'
        if (run) in myLumiDB:
            theLumi = myLumiDB[run]
            print("run",run," int. lumi:",theLumi,"/pb")
        else:
            print("=====> COULD NOT FIND LUMI, setting default = 1/pb")
            theLumi='1'
            print("run",run," int. lumi:",theLumi,"/pb")

        # loop on the dictionary
        for key, value in inputDict.items():            
            #print(key,value)
            if "Input" in key:
                continue
            else:
                key = key.split(":", 1)[1]
                print("dealing with",key)

            os.system("cp "+input_CMSSW_BASE+"/src/Alignment/OfflineValidation/test/PrimaryVertexResolution_templ_cfg.py ./cfg/PrimaryVertexResolution_"+key+"_"+run+"_cfg.py")
            os.system("sed -i 's|XXX_FILES_XXX|"+listOfFiles+"|g' "+cwd+"/cfg/PrimaryVertexResolution_"+key+"_"+run+"_cfg.py")
            os.system("sed -i 's|XXX_RUN_XXX|"+run+"|g' "+cwd+"/cfg/PrimaryVertexResolution_"+key+"_"+run+"_cfg.py")
            os.system("sed -i 's|YYY_KEY_YYY|"+key+"|g' "+cwd+"/cfg/PrimaryVertexResolution_"+key+"_"+run+"_cfg.py")

            scriptFileName = os.path.join(bashdir,"batchHarvester_"+key+"_"+str(count-1)+".sh")
            scriptFile = open(scriptFileName,'w')
            scriptFile.write(batchScriptCERN(input_CMSSW_BASE,run,eosdir,theLumi,key,value))
            scriptFile.close()
            #os.system('chmod +x %s' % scriptFileName)

    ## prepare the HTCondor submission files and eventually submit them
    for key, value in inputDict.items():
        if "Input" in key:
            continue
        else:
            key = key.split(":", 1)[1]

        job_submit_file = write_HTCondor_submit_file(bashdir,"batchHarvester_"+key,count,None)

        if opts.submit:
            os.system("chmod u+x "+bashdir+"/*.sh")
            submissionCommand = "condor_submit "+job_submit_file
            print(submissionCommand)
            os.system(submissionCommand)

###################################################
if __name__ == "__main__":        
    main()
