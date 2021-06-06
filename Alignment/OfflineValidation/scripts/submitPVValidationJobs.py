#!/usr/bin/env python

'''Script that submits CMS Tracker Alignment Primary Vertex Validation workflows,
usage:

submitPVValidationJobs.py -j TEST -D /HLTPhysics/Run2016C-TkAlMinBias-07Dec2018-v1/ALCARECO -i testPVValidation_Relvals_DATA.ini -r
'''

from __future__ import print_function
from builtins import range

__author__ = 'Marco Musich'
__copyright__ = 'Copyright 2020, CERN CMS'
__credits__ = ['Ernesto Migliore', 'Salvatore Di Guida']
__license__ = 'Unknown'
__maintainer__ = 'Marco Musich'
__email__ = 'marco.musich@cern.ch'
__version__ = 1

import datetime,time
import os,sys
import copy
import pickle
import string, re
import configparser as ConfigParser
import json
import pprint
import subprocess
from optparse import OptionParser
from subprocess import Popen, PIPE
import collections
import warnings
import shutil
import multiprocessing
from enum import Enum

class RefitType(Enum):
    STANDARD = 1
    COMMON   = 2

CopyRights  = '##################################\n'
CopyRights += '#    submitPVValidationJobs.py   #\n'
CopyRights += '#      marco.musich@cern.ch      #\n'
CopyRights += '#           April 2020           #\n'
CopyRights += '##################################\n'

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
requirements          = (OpSysAndVer =?= "CentOS7")
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
def getCommandOutput(command):
##############################################
    """This function executes `command` and returns it output.
    Arguments:
    - `command`: Shell command to be invoked by this function.
    """
    child = os.popen(command)
    data = child.read()
    err = child.close()
    if err:
        print('%s failed w/ exit code %d' % (command, err))
    return data

##############################################
def getFilesForRun(blob):
##############################################
    cmd2 = ' dasgoclient -limit=0 -query \'file run='+blob[0][0]+' dataset='+blob[0][1]+ (' instance='+blob[1]+'\'' if (blob[1] is not None) else '\'')
    #cmd2 = 'dasgoclient -query \'file run='+blob[0]+' dataset='+blob[1]+'\''
    q = Popen(cmd2 , shell=True, stdout=PIPE, stderr=PIPE)
    out, err = q.communicate()
    #print(cmd2,'\n',out.rstrip('\n'))
    outputList = out.decode().split('\n')
    outputList.pop()
    return outputList #,err

##############################################
def getNEvents(run, dataset):
##############################################
    nEvents = subprocess.check_output(["das_client", "--limit", "0", "--query", "summary run={} dataset={} | grep summary.nevents".format(run, dataset)])
    return 0 if nEvents == "[]\n" else int(nEvents)

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
        #output = subprocess.check_output([homedir+"/.local/bin/brilcalc", "lumi", "-b", "STABLE BEAMS", "--normtag=/afs/cern.ch/user/l/lumipro/public/normtag_file/normtag_BRIL.json", "-u", "/pb", "--begin", str(minRun),"--end",str(maxRun),"--output-style","csv"])

        output = subprocess.check_output([homedir+"/.local/bin/brilcalc", "lumi", "-b", "STABLE BEAMS","-u", "/pb", "--begin", str(minRun),"--end",str(maxRun),"--output-style","csv","-c","web"])
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
            #print "run",runToCache
            #print "lumi",lumiToCache
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

##############################################
def to_bool(value):
##############################################
    """
       Converts 'something' to boolean. Raises exception for invalid formats
           Possible True  values: 1, True, "1", "TRue", "yes", "y", "t"
           Possible False values: 0, False, None, [], {}, "", "0", "faLse", "no", "n", "f", 0.0, ...
    """
    if str(value).lower() in ("yes", "y", "true",  "t", "1"): return True
    if str(value).lower() in ("no",  "n", "false", "f", "0", "0.0", "", "none", "[]", "{}"): return False
    raise Exception('Invalid value for boolean conversion: ' + str(value))

##############################################
def updateDB2():
##############################################
    dbName = "runInfo.pkl"
    infos = {}
    if os.path.exists(dbName):
        with open(dbName,'rb') as f:
            infos = pickle.load(f)

    for f in glob.glob("root-files/Run*.root"):
        run = runFromFilename(f)
        if run not in infos:
            infos[run] = {}
            infos[run]["start_time"] = getRunStartTime(run)
            infos["isValid"] = isValid(f)

    with open(dbName, "wb") as f:
        pickle.dump(infos, f)

##############################################
def updateDB(run,runInfo):
##############################################
    dbName = "runInfo.pkl"
    infos = {}
    if os.path.exists(dbName):
        with open(dbName,'rb') as f:
            infos = pickle.load(f)

    if run not in infos:
        infos[run] = runInfo

    with open(dbName, "wb") as f:
        pickle.dump(infos, f)

####################--- Classes ---############################
class BetterConfigParser(ConfigParser.ConfigParser):

    ##############################################
    def optionxform(self, optionstr):
        return optionstr

    ##############################################
    def exists( self, section, option):
         try:
             items = self.items(section) 
         except ConfigParser.NoSectionError:
             return False
         for item in items:
             if item[0] == option:
                 return True
         return False

    ##############################################
    def __updateDict( self, dictionary, section ):
        result = dictionary
        try:
            for option in self.options( section ):
                result[option] = self.get( section, option )
            if "local"+section.title() in self.sections():
                for option in self.options( "local"+section.title() ):
                    result[option] = self.get( "local"+section.title(),option )
        except ConfigParser.NoSectionError as section:
            msg = ("%s in configuration files. This section is mandatory."
                   %(str(section).replace(":", "", 1)))
            #raise AllInOneError(msg)
        return result     

    ##############################################
    def getResultingSection( self, section, defaultDict = {}, demandPars = [] ):
        result = copy.deepcopy(defaultDict)
        for option in demandPars:
            try:
                result[option] = self.get( section, option )
            except ConfigParser.NoOptionError as globalSectionError:
                globalSection = str( globalSectionError ).split( "'" )[-2]
                splittedSectionName = section.split( ":" )
                if len( splittedSectionName ) > 1:
                    localSection = ("local"+section.split( ":" )[0].title()+":"
                                    +section.split(":")[1])
                else:
                    localSection = ("local"+section.split( ":" )[0].title())
                if self.has_section( localSection ):
                    try:
                        result[option] = self.get( localSection, option )
                    except ConfigParser.NoOptionError as option:
                        msg = ("%s. This option is mandatory."
                               %(str(option).replace(":", "", 1).replace(
                                   "section",
                                   "section '"+globalSection+"' or", 1)))
                        #raise AllInOneError(msg)
                else:
                    msg = ("%s. This option is mandatory."
                           %(str(globalSectionError).replace(":", "", 1)))
                    #raise AllInOneError(msg)
        result = self.__updateDict( result, section )
        #print(result)
        return result

##### method to parse the input file ################################
def ConfigSectionMap(config, section):
    the_dict = {}
    options = config.options(section)
    for option in options:
        try:
            the_dict[option] = config.get(section, option)
            if the_dict[option] == -1:
                DebugPrint("skip: %s" % option)
        except:
            print("exception on %s!" % option)
            the_dict[option] = None
    return the_dict

###### method to create recursively directories on EOS #############
def mkdir_eos(out_path):
    print("creating",out_path)
    newpath='/'
    for dir in out_path.split('/'):
        newpath=os.path.join(newpath,dir)
        # do not issue mkdir from very top of the tree
        if newpath.find('test_out') > 0:
            #getCommandOutput("eos mkdir"+newpath)
            command="/afs/cern.ch/project/eos/installation/cms/bin/eos.select mkdir "+newpath
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

def split(sequence, size):
##########################    
# aux generator function to split lists
# based on http://sandrotosi.blogspot.com/2011/04/python-group-list-in-sub-lists-of-n.html
# about generators see also http://stackoverflow.com/questions/231767/the-python-yield-keyword-explained
##########################
    for i in range(0, len(sequence), size):
        yield sequence[i:i+size] 

#############
class Job:
#############

    def __init__(self,dataset, job_number, job_id, job_name, isDA, isMC, applyBOWS, applyEXTRACOND, extraconditions, runboundary, lumilist, intlumi, maxevents, gt, allFromGT, alignmentDB, alignmentTAG, apeDB, apeTAG, bowDB, bowTAG, vertextype, tracktype, refittertype, ttrhtype, applyruncontrol, ptcut, CMSSW_dir ,the_dir):
###############################

        theDataSet = dataset.split("/")[1]+"_"+(dataset.split("/")[2]).split("-")[0]

        self.data              = theDataSet
        self.job_number        = job_number
        self.job_id            = job_id    
        self.batch_job_id      = None 
        self.job_name          = job_name
        
        self.isDA              = isDA             
        self.isMC              = isMC             
        self.applyBOWS         = applyBOWS
        self.applyEXTRACOND    = applyEXTRACOND
        self.extraCondVect     = extraconditions
        self.runboundary       = runboundary         
        self.lumilist          = lumilist        
        self.intlumi           = intlumi
        self.maxevents         = maxevents
        self.gt                = gt
        self.allFromGT         = allFromGT
        self.alignmentDB       = alignmentDB      
        self.alignmentTAG      = alignmentTAG     
        self.apeDB             = apeDB            
        self.apeTAG            = apeTAG           
        self.bowDB             = bowDB            
        self.bowTAG            = bowTAG           
        self.vertextype        = vertextype       
        self.tracktype         = tracktype
        self.refittertype      = refittertype
        self.ttrhtype          = ttrhtype
        self.applyruncontrol   = applyruncontrol  
        self.ptcut             = ptcut            

        self.the_dir=the_dir
        self.CMSSW_dir=CMSSW_dir

        self.output_full_name=self.getOutputBaseName()+"_"+str(self.job_id)
        self.output_number_name=self.getOutputBaseNameWithData()+"_"+str(self.job_number)
        
        self.cfg_dir=None
        self.outputCfgName=None
        
        # LSF variables        
        self.LSF_dir=None
        self.BASH_dir=None
        self.output_LSF_name=None
        self.output_BASH_name=None

        self.lfn_list=list()      

    def __del__(self):
###############################
        del self.lfn_list

    def setEOSout(self,theEOSdir):    
###############################
        self.OUTDIR = theEOSdir
          
    def getOutputBaseName(self):
########################    
        return "PVValidation_"+self.job_name

    def getOutputBaseNameWithData(self):
########################    
        return "PVValidation_"+self.job_name+"_"+self.data
        
    def createTheCfgFile(self,lfn):
###############################

        global CopyRights
        # write the cfg file

        self.cfg_dir = os.path.join(self.the_dir,"cfg")
        if not os.path.exists(self.cfg_dir):
            os.makedirs(self.cfg_dir)

        self.outputCfgName=self.output_full_name+"_cfg.py"
        fout=open(os.path.join(self.cfg_dir,self.outputCfgName),'w')

        template_cfg_file = os.path.join(self.the_dir,"PVValidation_T_cfg.py")

        fin = open(template_cfg_file)

        config_txt = '\n\n' + CopyRights + '\n\n'
        config_txt += fin.read()

        config_txt=config_txt.replace("ISDATEMPLATE",self.isDA)
        config_txt=config_txt.replace("ISMCTEMPLATE",self.isMC)
        config_txt=config_txt.replace("APPLYBOWSTEMPLATE",self.applyBOWS)
        config_txt=config_txt.replace("EXTRACONDTEMPLATE",self.applyEXTRACOND)
        config_txt=config_txt.replace("USEFILELISTTEMPLATE","True")
        config_txt=config_txt.replace("RUNBOUNDARYTEMPLATE",self.runboundary)
        config_txt=config_txt.replace("LUMILISTTEMPLATE",self.lumilist)
        config_txt=config_txt.replace("MAXEVENTSTEMPLATE",self.maxevents)
        config_txt=config_txt.replace("GLOBALTAGTEMPLATE",self.gt)
        config_txt=config_txt.replace("ALLFROMGTTEMPLATE",self.allFromGT)
        config_txt=config_txt.replace("ALIGNOBJTEMPLATE",self.alignmentDB)
        config_txt=config_txt.replace("GEOMTAGTEMPLATE",self.alignmentTAG)
        config_txt=config_txt.replace("APEOBJTEMPLATE",self.apeDB)
        config_txt=config_txt.replace("ERRORTAGTEMPLATE",self.apeTAG)
        config_txt=config_txt.replace("BOWSOBJECTTEMPLATE",self.bowDB)
        config_txt=config_txt.replace("BOWSTAGTEMPLATE",self.bowTAG)
        config_txt=config_txt.replace("VERTEXTYPETEMPLATE",self.vertextype)
        config_txt=config_txt.replace("TRACKTYPETEMPLATE",self.tracktype)
        config_txt=config_txt.replace("REFITTERTEMPLATE",self.refittertype)
        config_txt=config_txt.replace("TTRHBUILDERTEMPLATE",self.ttrhtype)
        config_txt=config_txt.replace("PTCUTTEMPLATE",self.ptcut)
        config_txt=config_txt.replace("INTLUMITEMPLATE",self.intlumi)
        config_txt=config_txt.replace("RUNCONTROLTEMPLATE",self.applyruncontrol)
        lfn_with_quotes = map(lambda x: "\'"+x+"\'",lfn)
        config_txt=config_txt.replace("FILESOURCETEMPLATE","["+",".join(lfn_with_quotes)+"]")
        config_txt=config_txt.replace("OUTFILETEMPLATE",self.output_full_name+".root")

        fout.write(config_txt)

        for line in fin.readlines():

            if 'END OF EXTRA CONDITIONS' in line:
                for element in self.extraCondVect :
                    if("Rcd" in element):
                        params = self.extraCondVect[element].split(',')

                        fout.write(" \n")
                        fout.write("     process.conditionsIn"+element+"= CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone( \n")
                        fout.write("          connect = cms.string('"+params[0]+"'), \n")
                        fout.write("          toGet = cms.VPSet(cms.PSet(record = cms.string('"+element+"'), \n")
                        fout.write("                                     tag = cms.string('"+params[1]+"'), \n")
                        if (len(params)>2):
                            fout.write("                                     label = cms.untracked.string('"+params[2]+"') \n")
                        fout.write("                                     ) \n")
                        fout.write("                            ) \n")
                        fout.write("          ) \n")
                        fout.write("     process.prefer_conditionsIn"+element+" = cms.ESPrefer(\"PoolDBESSource\", \"conditionsIn"+element[0]+"\") \n \n") 
            fout.write(line)
        fout.close()
                          
    def createTheLSFFile(self):
###############################

       # directory to store the LSF to be submitted
        self.LSF_dir = os.path.join(self.the_dir,"LSF")
        if not os.path.exists(self.LSF_dir):
            os.makedirs(self.LSF_dir)

        self.output_LSF_name=self.output_full_name+".lsf"
        fout=open(os.path.join(self.LSF_dir,self.output_LSF_name),'w')
    
        job_name = self.output_full_name

        log_dir = os.path.join(self.the_dir,"log")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        fout.write("#!/bin/sh \n") 
        fout.write("#BSUB -L /bin/sh\n")       
        fout.write("#BSUB -J "+job_name+"\n")
        fout.write("#BSUB -o "+os.path.join(log_dir,job_name+".log")+"\n")
        fout.write("#BSUB -q cmscaf1nd \n")
        fout.write("JobName="+job_name+" \n")
        fout.write("OUT_DIR="+self.OUTDIR+" \n")
        fout.write("LXBATCH_DIR=`pwd` \n") 
        fout.write("cd "+os.path.join(self.CMSSW_dir,"src")+" \n")
        fout.write("eval `scram runtime -sh` \n")
        fout.write("cd $LXBATCH_DIR \n") 
        fout.write("cmsRun "+os.path.join(self.cfg_dir,self.outputCfgName)+" \n")
        fout.write("ls -lh . \n")
        fout.write("for RootOutputFile in $(ls *root ); do xrdcp -f ${RootOutputFile} root://eoscms//eos/cms${OUT_DIR}/${RootOutputFile} ; done \n")
        fout.write("for TxtOutputFile in $(ls *txt ); do xrdcp -f ${TxtOutputFile}  root://eoscms//eos/cms${OUT_DIR}/${TxtOutputFile} ; done \n")

        fout.close()


    def createTheBashFile(self):
###############################

       # directory to store the BASH to be submitted
        self.BASH_dir = os.path.join(self.the_dir,"BASH")
        if not os.path.exists(self.BASH_dir):
            os.makedirs(self.BASH_dir)

        self.output_BASH_name=self.output_number_name+".sh"
        fout=open(os.path.join(self.BASH_dir,self.output_BASH_name),'w')
    
        job_name = self.output_full_name

        log_dir = os.path.join(self.the_dir,"log")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        fout.write("#!/bin/bash \n")
        #fout.write("export EOS_MGM_URL=root://eoscms.cern.ch \n")
        fout.write("JobName="+job_name+" \n")
        fout.write("echo  \"Job started at \" `date` \n")
        fout.write("CMSSW_DIR="+os.path.join(self.CMSSW_dir,"src")+" \n")
        fout.write("export X509_USER_PROXY=$CMSSW_DIR/Alignment/OfflineValidation/test/.user_proxy \n")
        fout.write("OUT_DIR="+self.OUTDIR+" \n")
        fout.write("LXBATCH_DIR=$PWD \n") 
        #fout.write("cd "+os.path.join(self.CMSSW_dir,"src")+" \n")
        fout.write("cd ${CMSSW_DIR} \n")
        fout.write("eval `scramv1 runtime -sh` \n")
        fout.write("echo \"batch dir: $LXBATCH_DIR release: $CMSSW_DIR release base: $CMSSW_RELEASE_BASE\" \n") 
        fout.write("cd $LXBATCH_DIR \n") 
        fout.write("cp "+os.path.join(self.cfg_dir,self.outputCfgName)+" . \n")
        fout.write("echo \"cmsRun "+self.outputCfgName+"\" \n")
        fout.write("cmsRun "+self.outputCfgName+" \n")
        fout.write("echo \"Content of working dir is \"`ls -lh` \n")
        #fout.write("less condor_exec.exe \n")
        fout.write("for RootOutputFile in $(ls *root ); do xrdcp -f ${RootOutputFile} root://eoscms//eos/cms${OUT_DIR}/${RootOutputFile} ; done \n")
        #fout.write("mv ${JobName}.out ${CMSSW_DIR}/BASH \n")
        fout.write("echo  \"Job ended at \" `date` \n")
        fout.write("exit 0 \n")

        fout.close()

    def getOutputFileName(self):
############################################
        return os.path.join(self.OUTDIR,self.output_full_name+".root")
        
    def submit(self):
###############################        
        print("submit job", self.job_id)
        job_name = self.output_full_name
        submitcommand1 = "chmod u+x " + os.path.join(self.LSF_dir,self.output_LSF_name)
        child1  = os.system(submitcommand1)
        #submitcommand2 = "bsub < "+os.path.join(self.LSF_dir,self.output_LSF_name)
        #child2  = os.system(submitcommand2)
        self.batch_job_id = getCommandOutput("bsub < "+os.path.join(self.LSF_dir,self.output_LSF_name))

    def getBatchjobId(self):    
############################################
       return self.batch_job_id.split("<")[1].split(">")[0] 

##############################################
def main():
##############################################

    ## check first there is a valid grid proxy
    if not check_proxy():
        print("Please create proxy via 'voms-proxy-init -voms cms -rfc'.")
        sys.exit(1)

    ## check first there is a valid grid proxy
    forward_proxy(".")

    global CopyRights
    print('\n'+CopyRights)

    HOME = os.environ.get('HOME')

    # CMSSW section
    input_CMSSW_BASE = os.environ.get('CMSSW_BASE')
    AnalysisStep_dir = os.path.join(input_CMSSW_BASE,"src/Alignment/OfflineValidation/test")
    lib_path = os.path.abspath(AnalysisStep_dir)
    sys.path.append(lib_path)

    ## N.B.: this is dediced here once and for all
    srcFiles        = []

    desc="""This is a description of %prog."""
    parser = OptionParser(description=desc,version='%prog version 0.1')
    parser.add_option('-s','--submit',    help='job submitted',         dest='submit',     action='store_true',  default=False)
    parser.add_option('-j','--jobname',   help='task name',             dest='taskname',   action='store',       default='myTask')
    parser.add_option('-D','--dataset',   help='selected dataset',      dest='data',       action='store',       default='')
    parser.add_option('-r','--doRunBased',help='selected dataset',      dest='doRunBased', action='store_true' , default=False)
    parser.add_option('-i','--input',     help='set input configuration (overrides default)', dest='inputconfig',action='store',default=None)
    parser.add_option('-b','--begin',     help='starting point',        dest='start',      action='store',       default='1')
    parser.add_option('-e','--end',       help='ending point',          dest='end',        action='store',       default='999999')
    parser.add_option('-v','--verbose',   help='verbose output',        dest='verbose',    action='store_true',  default=False)
    parser.add_option('-u','--unitTest',  help='unit tests?',           dest='isUnitTest', action='store_true',  default=False)
    parser.add_option('-I','--instance',  help='DAS instance to use',   dest='instance',   action='store',       default=None) 
    (opts, args) = parser.parse_args()

    now = datetime.datetime.now()
    #t = now.strftime("test_%Y_%m_%d_%H_%M_%S_")
    #t = "2016UltraLegacy"
    #t = "2017UltraLegacy"
    #t = "2018UltraLegacy"
    t=""
    t+=opts.taskname
    
    USER = os.environ.get('USER')
    eosdir=os.path.join("/store/group/alca_trackeralign",USER,"test_out",t)
    
    if opts.submit:
        mkdir_eos(eosdir)
    else:
        print("Not going to create EOS folder. -s option has not been chosen")

    #### Initialize all the variables

    jobName         = []
    isMC            = []
    isDA            = []
    doRunBased      = []
    maxevents       = []

    gt              = []
    allFromGT       = []
    applyEXTRACOND  = []
    extraCondVect   = []      
    alignmentDB     = []
    alignmentTAG    = []
    apeDB           = []
    apeTAG          = []
    applyBOWS       = []
    bowDB           = []
    bowTAG          = []
    conditions      = []
    
    vertextype      = []
    tracktype       = []
    refittertype    = []
    ttrhtype        = []

    applyruncontrol = []
    ptcut           = []
    runboundary     = []
    lumilist        = []
      
    ConfigFile = opts.inputconfig
    
    if ConfigFile is not None:

        print("********************************************************")
        print("* Parsing from input file:", ConfigFile," ")
        
        config = BetterConfigParser()
        config.read(ConfigFile)

        print("Parsed the following configuration \n\n")
        inputDict = as_dict(config)
        pprint.pprint(inputDict)

        if(not bool(inputDict)):
            raise SystemExit("\n\n ERROR! Could not parse any input file, perhaps you are submitting this from the wrong folder? \n\n")

        #print  config.sections()

        # please notice: since in principle one wants to run on several different samples simultaneously,
        # all these inputs are vectors

        doRunBased       = opts.doRunBased

        listOfValidations = config.getResultingSection("validations")
        
        for item in listOfValidations:
            if (bool(listOfValidations[item]) == True):
                
                jobName.append(ConfigSectionMap(config,"Conditions:"+item)['jobname'])
                isDA.append(ConfigSectionMap(config,"Job")['isda'])
                isMC.append(ConfigSectionMap(config,"Job")['ismc'])
                maxevents.append(ConfigSectionMap(config,"Job")['maxevents'])

                gt.append(ConfigSectionMap(config,"Conditions:"+item)['gt'])
                allFromGT.append(ConfigSectionMap(config,"Conditions:"+item)['allFromGT'])
                applyEXTRACOND.append(ConfigSectionMap(config,"Conditions:"+item)['applyextracond'])
                conditions.append(config.getResultingSection("ExtraConditions"))
                
                alignmentDB.append(ConfigSectionMap(config,"Conditions:"+item)['alignmentdb'])
                alignmentTAG.append(ConfigSectionMap(config,"Conditions:"+item)['alignmenttag'])
                apeDB.append(ConfigSectionMap(config,"Conditions:"+item)['apedb'])
                apeTAG.append(ConfigSectionMap(config,"Conditions:"+item)['apetag'])
                applyBOWS.append(ConfigSectionMap(config,"Conditions:"+item)['applybows'])
                bowDB.append(ConfigSectionMap(config,"Conditions:"+item)['bowdb'])
                bowTAG.append(ConfigSectionMap(config,"Conditions:"+item)['bowtag'])
                
                vertextype.append(ConfigSectionMap(config,"Type")['vertextype'])     
                tracktype.append(ConfigSectionMap(config,"Type")['tracktype'])

                ## in case there exists a specification for the refitter

                if(config.exists("Refit","refittertype")):
                    refittertype.append(ConfigSectionMap(config,"Refit")['refittertype'])
                else:
                    refittertype.append(str(RefitType.COMMON))

                if(config.exists("Refit","ttrhtype")):
                    ttrhtype.append(ConfigSectionMap(config,"Refit")['ttrhtype'])
                else:
                    ttrhtype.append("WithAngleAndTemplate")

                applyruncontrol.append(ConfigSectionMap(config,"Selection")['applyruncontrol'])
                ptcut.append(ConfigSectionMap(config,"Selection")['ptcut'])
                runboundary.append(ConfigSectionMap(config,"Selection")['runboundary'])
                lumilist.append(ConfigSectionMap(config,"Selection")['lumilist'])
    else :

        print("********************************************************")
        print("* Parsing from command line                            *")
        print("********************************************************")
          
        jobName         = ['testing']
        isDA            = ['True']   
        isMC            = ['True']
        doRunBased      = opts.doRunBased
        maxevents       = ['10000']
        
        gt              = ['74X_dataRun2_Prompt_v4']       
        allFromGT       = ['False']
        applyEXTRACOND  = ['False']
        conditions      = [[('SiPixelTemplateDBObjectRcd','frontier://FrontierProd/CMS_CONDITIONS','SiPixelTemplateDBObject_38T_2015_v3_hltvalidation')]]
        alignmentDB     = ['frontier://FrontierProd/CMS_CONDITIONS']
        alignmentTAG    = ['TrackerAlignment_Prompt']  
        apeDB           = ['frontier://FrontierProd/CMS_CONDITIONS']  
        apeTAG          = ['TrackerAlignmentExtendedErr_2009_v2_express_IOVs']
        applyBOWS       = ['True']  
        bowDB           = ['frontier://FrontierProd/CMS_CONDITIONS']  
        bowTAG          = ['TrackerSurafceDeformations_v1_express']  
        
        vertextype      = ['offlinePrimaryVertices']
        tracktype       = ['ALCARECOTkAlMinBias']

        applyruncontrol = ['False']  
        ptcut           = ['3'] 
        runboundary     = ['1']  
        lumilist        = ['']  
 
    # print some of the configuration
    
    print("********************************************************")
    print("* Configuration info *")
    print("********************************************************")
    print("- submitted   : ",opts.submit)
    print("- taskname    : ",opts.taskname)
    print("- Jobname     : ",jobName)
    print("- use DA      : ",isDA)
    print("- is MC       : ",isMC)
    print("- is run-based: ",doRunBased)
    print("- evts/job    : ",maxevents)
    print("- GlobatTag   : ",gt)
    print("- allFromGT?  : ",allFromGT)
    print("- extraCond?  : ",applyEXTRACOND)
    print("- extraCond   : ",conditions)
    print("- Align db    : ",alignmentDB)
    print("- Align tag   : ",alignmentTAG)
    print("- APE db      : ",apeDB)
    print("- APE tag     : ",apeTAG)
    print("- use bows?   : ",applyBOWS)
    print("- K&B db      : ",bowDB)
    print("- K&B tag     : ",bowTAG)
    print("- VertexColl  : ",vertextype)
    print("- TrackColl   : ",tracktype)
    print("- RefitterSeq : ",refittertype)
    print("- TTRHBuilder : ",ttrhtype)
    print("- RunControl? : ",applyruncontrol) 
    print("- Pt>           ",ptcut)
    print("- run=          ",runboundary)
    print("- JSON        : ",lumilist)
    print("- Out Dir     : ",eosdir)

    print("********************************************************")
    print("Will run on",len(jobName),"workflows")

    myRuns = []
    mylist = {}

    if(doRunBased):
        print(">>>> This is Data!")
        print(">>>> Doing run based selection")
        cmd = 'dasgoclient -limit=0 -query \'run dataset='+opts.data + (' instance='+opts.instance+'\'' if (opts.instance is not None) else '\'')
        p = Popen(cmd , shell=True, stdout=PIPE, stderr=PIPE)
        out, err = p.communicate()
        #print(out)
        listOfRuns=out.decode().split("\n")
        listOfRuns.pop()
        listOfRuns.sort()
        print("Will run on ",len(listOfRuns),"runs: \n",listOfRuns)

        mytuple=[]

        print("first run:",opts.start,"last run:",opts.end)

        for run in listOfRuns:
            if (int(run)<int(opts.start) or int(run)>int(opts.end)):
                print("excluding",run)
                continue

            if not isInJSON(run,lumilist[0]):
                continue

            else:
                print("'======> taking",run)
            #print "preparing run",run
            #if(int(run)%100==0):
            mytuple.append((run,opts.data))

        #print mytuple

        instances=[opts.instance for entry in mytuple]
        pool = multiprocessing.Pool(processes=20)  # start 20 worker processes
        count = pool.map(getFilesForRun,zip(mytuple,instances))
        file_info = dict(zip(listOfRuns, count))

        #print file_info

        for run in listOfRuns:
            if (int(run)<int(opts.start) or int(run)>int(opts.end)):
                print('rejecting run',run,' becasue outside of boundaries')
                continue

            if not isInJSON(run,lumilist[0]):
                print('rejecting run',run,' becasue outside not in JSON')
                continue

            #if(int(run)%100==0):
            #    print "preparing run",run
            myRuns.append(run)
            #cmd2 = ' das_client --limit=0 --query \'file run='+run+' dataset='+opts.data+'\''
            #q = Popen(cmd2 , shell=True, stdout=PIPE, stderr=PIPE)
            #out2, err2 = q.communicate()
        
            #out2=getFilesForRun((run,opts.data))
            #print out2
            #pool.map(getFilesForRun,run,opts.data)


            #if run in file_info:
                #mylist[run] = file_info[run]
                #print run,mylist[run]
            #mylist[run] = out2.split('\n')
            #print mylist
            #mylist[run].pop()
            #print mylist

        od = collections.OrderedDict(sorted(file_info.items()))
        # print od

        ## check that the list of runs is not empty
        if(len(myRuns)==0):
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
            myLumiDB = getLuminosity(HOME,myRuns[0],myRuns[-1],doRunBased,opts.verbose)

        if(opts.verbose):
            pprint.pprint(myLumiDB)

    # start loop on samples
    for iConf in range(len(jobName)):
        print("This is Task n.",iConf+1,"of",len(jobName))
        
        ##  print "==========>",conditions

        # for hadd script
        scripts_dir = os.path.join(AnalysisStep_dir,"scripts")
        if not os.path.exists(scripts_dir):
            os.makedirs(scripts_dir)
        hadd_script_file = os.path.join(scripts_dir,jobName[iConf]+"_"+opts.taskname+".sh")
        fout = open(hadd_script_file,'w')

        output_file_list1=list()      
        output_file_list2=list()
        output_file_list2.append("hadd ")
              
        inputFiles = []

        if (to_bool(isMC[iConf]) or (not to_bool(doRunBased))):
            if(to_bool(isMC[iConf])):
                print("this is MC")
                cmd = 'dasgoclient -query \'file dataset='+opts.data+ (' instance='+opts.instance+'\'' if (opts.instance is not None) else '\'')
                s = Popen(cmd , shell=True, stdout=PIPE, stderr=PIPE)
                out,err = s.communicate()
                mylist = out.decode().split('\n')
                mylist.pop()
                #print mylist
           
                splitList = split(mylist,10)
                for files in splitList:
                    inputFiles.append(files)
                    myRuns.append(str(1))
            else:
                print("this is DATA (not doing full run-based selection)")
                print(runboundary[iConf])
                cmd = 'dasgoclient -query \'file dataset='+opts.data+' run='+runboundary[iConf]+ (' instance='+opts.instance+'\'' if (opts.instance is not None) else '\'')
                #print cmd
                s = Popen(cmd , shell=True, stdout=PIPE, stderr=PIPE)
                out,err = s.communicate()
                #print(out)
                mylist = out.decode().split('\n')
                mylist.pop()
                #print "len(mylist):",len(mylist)
                print("mylist:",mylist)

                splitList = split(mylist,10)
                for files in splitList:
                    inputFiles.append(files)
                    myRuns.append(str(runboundary[iConf]))

                myLumiDB = getLuminosity(HOME,myRuns[0],myRuns[-1],True,opts.verbose)

        else:
            #pass
            for element in od:
                #print mylist[element]
                inputFiles.append(od[element])
                #print element,od[element]
            #print mylist

        #print inputFiles

        ## declare here the list of jobs that should be waited for
        batchJobIds = []
        mergedFile = None

        if(opts.verbose):
            print("myRuns =====>",myRuns)

        totalJobs=0
        theBashDir=None
        theBaseName=None

        for jobN,theSrcFiles in enumerate(inputFiles):
            if(opts.verbose):
                print("JOB:",jobN,"run",myRuns[jobN],theSrcFiles)
            else:
                print("JOB:",jobN,"run",myRuns[jobN])
            thejobIndex=None
            theLumi='1'

            #if(to_bool(isMC[iConf]) and (not to_bool(doRunBased))):
            if(to_bool(isMC[iConf])):
                thejobIndex=jobN
            else:
                if(doRunBased):
                    thejobIndex=myRuns[jobN]
                else:
                    thejobIndex=myRuns[jobN]+"_"+str(jobN)

                if (myRuns[jobN]) in myLumiDB:
                    theLumi = myLumiDB[myRuns[jobN]]
                else:
                    print("=====> COULD NOT FIND LUMI, setting default = 1/pb")
                    theLumi='1'
                print("int. lumi:",theLumi,"/pb")

            #print 'the configuration is:',iConf,' theJobIndex is:',thejobIndex
            #print applyBOWS[iConf],applyEXTRACOND[iConf],conditions[iConf]

            runInfo = {}
            runInfo["run"]             = myRuns[jobN]
            #runInfo["runevents"]      = getNEvents(myRuns[jobN],opts.data) 
            runInfo["conf"]            = jobName[iConf]
            runInfo["gt"]              = gt[iConf]
            runInfo["allFromGT"]       = allFromGT[iConf] 
            runInfo["alignmentDB"]     = alignmentDB[iConf]
            runInfo["alignmentTag"]    = alignmentTAG[iConf]
            runInfo["apeDB"]           = apeDB[iConf]
            runInfo["apeTag"]          = apeTAG[iConf]
            runInfo["applyBows"]       = applyBOWS[iConf]
            runInfo["bowDB"]           = bowDB[iConf]
            runInfo["bowTag"]          = bowTAG[iConf]
            runInfo["ptCut"]           = ptcut[iConf]
            runInfo["lumilist"]        = lumilist[iConf]
            runInfo["applyEXTRACOND"]  = applyEXTRACOND[iConf]
            runInfo["conditions"]      = conditions[iConf]
            runInfo["nfiles"]          = len(theSrcFiles)
            runInfo["srcFiles"]        = theSrcFiles
            runInfo["intLumi"]         = theLumi

            updateDB(((iConf+1)*10)+(jobN+1),runInfo)

            totalJobs=totalJobs+1

            aJob = Job(opts.data,
                       jobN,
                       thejobIndex,
                       jobName[iConf],isDA[iConf],isMC[iConf],
                       applyBOWS[iConf],applyEXTRACOND[iConf],conditions[iConf],
                       myRuns[jobN], lumilist[iConf], theLumi, maxevents[iConf],
                       gt[iConf],allFromGT[iConf],
                       alignmentDB[iConf], alignmentTAG[iConf],
                       apeDB[iConf], apeTAG[iConf],
                       bowDB[iConf], bowTAG[iConf],
                       vertextype[iConf], tracktype[iConf],
                       refittertype[iConf], ttrhtype[iConf],
                       applyruncontrol[iConf],
                       ptcut[iConf],input_CMSSW_BASE,AnalysisStep_dir)
            
            aJob.setEOSout(eosdir)
            aJob.createTheCfgFile(theSrcFiles)
            aJob.createTheBashFile()

            output_file_list1.append("xrdcp root://eoscms//eos/cms"+aJob.getOutputFileName()+" /tmp/$USER/"+opts.taskname+" \n")
            if jobN == 0:
                theBashDir=aJob.BASH_dir
                theBaseName=aJob.getOutputBaseNameWithData()
                mergedFile = "/tmp/$USER/"+opts.taskname+"/"+aJob.getOutputBaseName()+" "+opts.taskname+".root"
                output_file_list2.append("/tmp/$USER/"+opts.taskname+"/"+aJob.getOutputBaseName()+opts.taskname+".root ")
            output_file_list2.append("/tmp/$USER/"+opts.taskname+"/"+os.path.split(aJob.getOutputFileName())[1]+" ")       
            del aJob

        job_submit_file = write_HTCondor_submit_file(theBashDir,theBaseName,totalJobs,None)

        if opts.submit:
            os.system("chmod u+x "+theBashDir+"/*.sh")
            submissionCommand = "condor_submit "+job_submit_file
            submissionOutput = getCommandOutput(submissionCommand)
            print(submissionOutput)

        fout.write("#!/bin/bash \n")
        fout.write("MAIL=$USER@mail.cern.ch \n")
        fout.write("OUT_DIR="+eosdir+"\n")
        fout.write("FILE="+str(mergedFile)+"\n")
        fout.write("echo $HOST | mail -s \"Harvesting job started\" $USER@mail.cern.ch \n")
        fout.write("cd "+os.path.join(input_CMSSW_BASE,"src")+"\n")
        fout.write("eval `scram r -sh` \n")
        fout.write("mkdir -p /tmp/$USER/"+opts.taskname+" \n")
        fout.writelines(output_file_list1)
        fout.writelines(output_file_list2)
        fout.write("\n")
        fout.write("echo \"xrdcp -f $FILE root://eoscms//eos/cms$OUT_DIR\" \n")
        fout.write("xrdcp -f $FILE root://eoscms//eos/cms$OUT_DIR \n")
        fout.write("echo \"Harvesting for "+opts.taskname+" task is complete; please find output at $OUT_DIR \" | mail -s \"Harvesting for " +opts.taskname +" completed\" $MAIL \n")

        os.system("chmod u+x "+hadd_script_file)

        harvest_conditions = '"' + " && ".join(["ended(" + jobId + ")" for jobId in batchJobIds]) + '"'
        print(harvest_conditions)
        lastJobCommand = "bsub -o harvester"+opts.taskname+".tmp -q 1nh -w "+harvest_conditions+" "+hadd_script_file
        print(lastJobCommand)
        if opts.submit:
            lastJobOutput = getCommandOutput(lastJobCommand)
            print(lastJobOutput)

            fout.close()
        del output_file_list1
        
###################################################
if __name__ == "__main__":        
    main()


   
