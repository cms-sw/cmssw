#!/usr/bin/env python

'''Script that submits CMS Tracker Alignment Primary Vertex Validation workflows
'''
from __future__ import print_function

from builtins import range
__author__ = 'Marco Musich'
__copyright__ = 'Copyright 2015, CERN CMS'
__credits__ = ['Ernesto Migliore', 'Salvatore Di Guida', 'Javier Duarte']
__license__ = 'Unknown'
__maintainer__ = 'Marco Musich'
__email__ = 'marco.musich@cern.ch'
__version__ = 1

import datetime,time
import os,sys
import copy
import string, re
import configparser as ConfigParser, json
from optparse import OptionParser
from subprocess import Popen, PIPE

CopyRights  = '##################################\n'
CopyRights += '#      submitAllJobs Script      #\n'
CopyRights += '#      marco.musich@cern.ch      #\n'
CopyRights += '#         December 2015          #\n'
CopyRights += '##################################\n'

##############################################
def drawProgressBar(percent, barLen=40):
##############################################
    sys.stdout.write("\r")
    progress = ""
    for i in range(barLen):
        if i < int(barLen * percent):
            progress += "="
        else:
            progress += " "
    sys.stdout.write("[ %s ] %.2f%%" % (progress, percent * 100))
    sys.stdout.flush()

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
        #print result
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
    newpath='/'
    for dir in out_path.split('/'):
        newpath=os.path.join(newpath,dir)
        # do not issue mkdir from very top of the tree
        if newpath.find('test_out') > 0:
            p = subprocess.Popen(["eos", "mkdir", newpath], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (out, err) = p.communicate()
            p.wait()

    # now check that the directory exists
    p = subprocess.Popen(["eos", "ls", out_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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

    def __init__(self, job_id, job_name, isDA, isMC, applyBOWS, applyEXTRACOND, extraconditions, runboundary, lumilist, maxevents, gt, allFromGT, alignmentDB, alignmentTAG, apeDB, apeTAG, bowDB, bowTAG, vertextype, tracktype, applyruncontrol, ptcut, CMSSW_dir ,the_dir):
###############################
        self.job_id=job_id    
        self.batch_job_id = None 
        self.job_name=job_name
        
        self.isDA              = isDA             
        self.isMC              = isMC             
        self.applyBOWS         = applyBOWS
        self.applyEXTRACOND    = applyEXTRACOND
        self.extraCondVect     = extraconditions
        self.runboundary       = runboundary         
        self.lumilist          = lumilist         
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
        self.applyruncontrol   = applyruncontrol  
        self.ptcut             = ptcut            

        self.the_dir=the_dir
        self.CMSSW_dir=CMSSW_dir

        self.output_full_name=self.getOutputBaseName()+"_"+str(self.job_id)

        self.cfg_dir=None
        self.outputCfgName=None
        
        # LSF variables        
        self.LSF_dir=None
        self.output_LSF_name=None

        self.lfn_list=list()      

        #self.OUTDIR = "" # TODO: write a setter method
        #self.OUTDIR = self.createEOSout()

    def __del__(self):
###############################
        del self.lfn_list

    def setEOSout(self,theEOSdir):    
###############################
        self.OUTDIR = theEOSdir
          
    def getOutputBaseName(self):
########################    
        return "PVValidation_"+self.job_name
        
    def createTheCfgFile(self,lfn):
###############################

        global CopyRights

        # write the cfg file 
        self.cfg_dir = os.path.join(self.the_dir,"cfg")
        if not os.path.exists(self.cfg_dir):
            os.makedirs(self.cfg_dir)

        self.outputCfgName=self.output_full_name+"_cfg.py"
        fout=open(os.path.join(self.cfg_dir,self.outputCfgName),'w+b')

        # decide which template according to data/mc
        if self.isMC:
            template_cfg_file = os.path.join(self.the_dir,"PVValidation_TEMPL_cfg.py")
        else:
            template_cfg_file = os.path.join(self.the_dir,"PVValidation_TEMPL_cfg.py")

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
        config_txt=config_txt.replace("PTCUTTEMPLATE",self.ptcut)               
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
                            fout.write("                                     label = cms.string('"+params[2]+"') \n")
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
        fout.write("for RootOutputFile in $(ls *root ); do xrdcp -f ${RootOutputFile}  root://eoscms//eos/cms${OUT_DIR}/${RootOutputFile} ; done \n")
        fout.write("for TxtOutputFile in $(ls *txt ); do xrdcp -f ${TxtOutputFile}  root://eoscms//eos/cms${OUT_DIR}/${TxtOutputFile} ; done \n")

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

    global CopyRights
    print('\n'+CopyRights)

    # CMSSW section
    input_CMSSW_BASE = os.environ.get('CMSSW_BASE')
    AnalysisStep_dir = os.path.join(input_CMSSW_BASE,"src/Alignment/OfflineValidation/test")
    sourceModule     = os.path.join(input_CMSSW_BASE,"src/Alignment/OfflineValidation/test","PVValidation_HLTPhysics2015B_TkAlMinBias_cff.py")
    lib_path = os.path.abspath(AnalysisStep_dir)
    sys.path.append(lib_path)

    ## N.B.: this is dediced here once and for all
    srcFiles        = []

    desc="""This is a description of %prog."""
    parser = OptionParser(description=desc,version='%prog version 0.1')
    parser.add_option('-s','--submit',    help='job submitted',    dest='submit',     action='store_true',  default=False)
    parser.add_option('-j','--jobname',   help='task name',        dest='taskname',   action='store',       default='')
    parser.add_option('-D','--dataset',   help='selected dataset', dest='data',       action='store'      , default='')
    parser.add_option('-r','--doRunBased',help='selected dataset', dest='doRunBased', action='store_true' , default=False)
    parser.add_option('-i','--input',     help='set input configuration (overrides default)', dest='inputconfig',action='store',default=None)
   
    (opts, args) = parser.parse_args()

    now = datetime.datetime.now()
    t = now.strftime("test_%Y_%m_%d_%H_%M_%S_DATA_")
    t+=opts.taskname
    
    USER = os.environ.get('USER')
    eosdir=os.path.join("/store/caf/user",USER,"test_out",t)
    #mkdir_eos(eosdir)

    #### Initialize all the variables

    jobName         = None
    isMC            = None
    isDA            = None
    doRunBased      = False
    maxevents       = None

    gt              = None
    allFromGT       = None
    applyEXTRACOND  = None
    extraCondVect   = None      
    alignmentDB     = None
    alignmentTAG    = None
    apeDB           = None
    apeTAG          = None
    applyBOWS       = None
    bowDB           = None
    bowTAG          = None

    vertextype      = None
    tracktype       = None

    applyruncontrol = None
    ptcut           = None
    runboundary     = None
    lumilist        = None
      
    ConfigFile = opts.inputconfig
    
    if ConfigFile is not None:

        print("********************************************************")
        print("* Parsing from input file:", ConfigFile," ")
        
        #config = ConfigParser.ConfigParser()
        #config.read(ConfigFile)

        config = BetterConfigParser()
        config.read(ConfigFile)

        #print  config.sections()

        # please notice: since in principle one wants to run on several different samples simultaneously,
        # all these inputs are vectors

        jobName          = [ConfigSectionMap(config,"Job")['jobname']]
        isDA             = [ConfigSectionMap(config,"Job")['isda']]
        isMC             = [ConfigSectionMap(config,"Job")['ismc']]
        doRunBased       = opts.doRunBased
        maxevents        = [ConfigSectionMap(config,"Job")['maxevents']]

        gt               = [ConfigSectionMap(config,"Conditions")['gt']]
        allFromGT        = [ConfigSectionMap(config,"Conditions")['allFromGT']]
        applyEXTRACOND   = [ConfigSectionMap(config,"Conditions")['applyextracond']]
        conditions       = [config.getResultingSection("ExtraConditions")]

        alignmentDB      = [ConfigSectionMap(config,"Conditions")['alignmentdb']]
        alignmentTAG     = [ConfigSectionMap(config,"Conditions")['alignmenttag']]
        apeDB            = [ConfigSectionMap(config,"Conditions")['apedb']]
        apeTAG           = [ConfigSectionMap(config,"Conditions")['apetag']]
        applyBOWS        = [ConfigSectionMap(config,"Conditions")['applybows']]
        bowDB            = [ConfigSectionMap(config,"Conditions")['bowdb']]
        bowTAG           = [ConfigSectionMap(config,"Conditions")['bowtag']]
        
        vertextype       = [ConfigSectionMap(config,"Type")['vertextype']]      
        tracktype        = [ConfigSectionMap(config,"Type")['tracktype']]
    
        applyruncontrol  = [ConfigSectionMap(config,"Selection")['applyruncontrol']]
        ptcut            = [ConfigSectionMap(config,"Selection")['ptcut']]
        runboundary      = [ConfigSectionMap(config,"Selection")['runboundary']]
        lumilist         = [ConfigSectionMap(config,"Selection")['lumilist']]
                      
    else :

        print("********************************************************")
        print("* Parsing from command line                            *")
        print("********************************************************")
          
        jobName         = ['MinBiasQCD_CSA14Ali_CSA14APE']
        isDA            = ['True']   
        isMC            = ['True']
        doRunBased      = opts.doRunBased
        maxevents       = ['10000']
        
        gt              = ['START53_V7A::All']       
        allFromGT       = ['False']
        applyEXTRACOND  = ['False']
        conditions      = [[('SiPixelTemplateDBObjectRcd','frontier://FrontierProd/CMS_COND_31X_PIXEL','SiPixelTemplates38T_2010_2011_mc'),
                            ('SiPixelQualityFromDBRcd','frontier://FrontierProd/CMS_COND_31X_PIXEL','SiPixelQuality_v20_mc')]]
        alignmentDB     = ['sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/PayLoads/TkAl-14-02_CSA14/Alignments_CSA14_v1.db']
        alignmentTAG    = ['TrackerCSA14Scenario']  
        apeDB           = ['sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/PayLoads/TkAl-14-02_CSA14/AlignmentErrors_CSA14_v1.db']  
        apeTAG          = ['TrackerCSA14ScenarioErrors']
        applyBOWS       = ['True']  
        bowDB           = ['frontier://FrontierProd/CMS_COND_310X_ALIGN']  
        bowTAG          = ['TrackerSurfaceDeformations_2011Realistic_v2_mc']  
        
        vertextype      = ['offlinePrimaryVertices']  
        tracktype       = ['ALCARECOTkAlMinBias']  
        
        applyruncontrol = ['False']  
        ptcut           = ['3'] 
        runboundary     = ['1']  
        lumilist        = ['']  
 
    # start loop on samples

    # print some of the configuration
    
    print("********************************************************")
    print("* Configuration info *")
    print("********************************************************")
    print("- submitted   : ",opts.submit)
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
    print("- RunControl? : ",applyruncontrol) 
    print("- Pt>           ",ptcut)           
    print("- run=          ",runboundary)     
    print("- JSON        : ",lumilist)        
    print("********************************************************")

    sublogging_dir = os.path.join(AnalysisStep_dir,"submissions")
    if not os.path.exists(sublogging_dir):
        os.makedirs(sublogging_dir)
    submission_log_file = os.path.join(sublogging_dir,"sub"+t+".log")
    log_fout = open(submission_log_file,'w')
    for iConf in range(len(jobName)):
        log_fout.write("============================================================ \n")
        log_fout.write("- timestamp   : "+t.strip("test_")+"\n")
        log_fout.write("- submitted   : "+str(opts.submit)+"\n")
        log_fout.write("- Jobname     : "+jobName[iConf]+"\n")           
        log_fout.write("- use DA      : "+isDA[iConf]+"\n")            
        log_fout.write("- is MC       : "+isMC[iConf]+"\n")            
        log_fout.write("- is run-based: "+str(doRunBased)+"\n")
        log_fout.write("- evts/job    : "+maxevents[iConf]+"\n")                    
        log_fout.write("- GlobatTag   : "+gt[iConf]+"\n")      
        log_fout.write("- allFromGT?  : "+allFromGT[iConf]+"\n")
        log_fout.write("- extraCond?  : "+applyEXTRACOND[iConf]+"\n")
        for x in conditions:
            for attribute,value in x.items():
                     log_fout.write('   - {} : {}'.format(attribute, value)+"\n")
        log_fout.write("- Align db    : "+alignmentDB[iConf]+"\n")     
        log_fout.write("- Align tag   : "+alignmentTAG[iConf]+"\n")    
        log_fout.write("- APE db      : "+apeDB[iConf]+"\n")           
        log_fout.write("- APE tag     : "+apeTAG[iConf]+"\n")          
        log_fout.write("- use bows?   : "+applyBOWS[iConf]+"\n")       
        log_fout.write("- K&B db      : "+bowDB[iConf]+"\n")
        log_fout.write("- K&B tag     : "+bowTAG[iConf]+"\n")                        
        log_fout.write("- VertexColl  : "+vertextype[iConf]+"\n")      
        log_fout.write("- TrackColl   : "+tracktype[iConf]+"\n")                       
        log_fout.write("- RunControl? : "+applyruncontrol[iConf]+"\n") 
        log_fout.write("- Pt>           "+ptcut[iConf]+"\n")           
        log_fout.write("- run=          "+runboundary[iConf]+"\n")     
        log_fout.write("- JSON        : "+lumilist[iConf]+"\n")
        log_fout.write("- output EOS  : "+eosdir+"\n")

    print("Will run on ",len(jobName),"workflows")

    for iConf in range(len(jobName)):
        print("Preparing",iConf," configurtion to run")

        # for hadd script
        scripts_dir = os.path.join(AnalysisStep_dir,"scripts")
        if not os.path.exists(scripts_dir):
            os.makedirs(scripts_dir)
        hadd_script_file = os.path.join(scripts_dir,jobName[iConf]+".sh")
        fout = open(hadd_script_file,'w')

        output_file_list1=list()      
        output_file_list2=list()
        output_file_list2.append("hadd ")
            
        inputFiles = []
        myRuns = []
        
        if (to_bool(isMC[iConf]) or (not to_bool(doRunBased))):
            if(to_bool(isMC[iConf])):
                print("this is MC")
                cmd = 'das_client.py --limit=0 --query \'file dataset='+opts.data+'\''
                s = Popen(cmd , shell=True, stdout=PIPE, stderr=PIPE)
                out,err = s.communicate()
                mylist = out.split('\n')
                mylist.pop()
                #print mylist
           
                splitList = split(mylist,10)
                for files in splitList:
                    inputFiles.append(files)
                    myRuns.append(str(1))
            else:
                print("this is DATA (not doing full run-based selection)")
                cmd = 'das_client.py --limit=0 --query \'file dataset='+opts.data+' run='+runboundary[iConf]+'\''
                #print cmd
                s = Popen(cmd , shell=True, stdout=PIPE, stderr=PIPE)
                out,err = s.communicate()
                mylist = out.split('\n')
                mylist.pop()
                #print "len(mylist):",len(mylist)
                print("mylist:",mylist)
                inputFiles.append(mylist)
                myRuns.append(str(runboundary[iConf]))

        else:
            print("this is Data")
            print("doing run based selection")
            cmd = 'das_client.py --limit=0 --query \'run dataset='+opts.data+'\''
            p = Popen(cmd , shell=True, stdout=PIPE, stderr=PIPE)
            out, err = p.communicate()
            listOfRuns=out.split('\n')
            listOfRuns.pop()
            listOfRuns.sort()
            myRuns = listOfRuns
            print("Will run on ",len(listOfRuns), " runs")
            print(listOfRuns)

            procs = []

            for run in listOfRuns:
                #print "preparing run",run
                cmd2 = ' das_client.py --limit=0 --query \'file run='+run+' dataset='+opts.data+'\''
                q = Popen(cmd2 , shell=True, stdout=PIPE, stderr=PIPE)
                procs.append(q)
                #out2, err2 = q.communicate()
                #mylist = out2.split('\n')
                #mylist.pop()
                #inputFiles.append(mylist)
                
            toolbar_width = len(listOfRuns)
            # setup toolbar
            print("********************************************************")
            print(" Retrieving run info")
            
            for i,p in enumerate(procs):
                out2,err2 = p.communicate()
                mylist = out2.split('\n')
                mylist.pop()
                inputFiles.append(mylist)
                #sys.stdout.write("-")
                #sys.stdout.flush()
                percent = float(i)/len(procs)
                #print percent
                drawProgressBar(percent)

            sys.stdout.write("\n") 

        for jobN,theSrcFiles in enumerate(inputFiles):
            print(jobN,"run",myRuns[jobN],theSrcFiles)
            thejobIndex=None
            batchJobIds = []

            #if(to_bool(isMC[iConf]) and (not to_bool(doRunBased))):
            if(to_bool(isMC[iConf])):
                thejobIndex=jobN
            else:
                thejobIndex=myRuns[jobN]

            aJob = Job(thejobIndex,
                       jobName[iConf],isDA[iConf],isMC[iConf],
                       applyBOWS[iConf],applyEXTRACOND[iConf],conditions[iConf],
                       myRuns[jobN], lumilist[iConf], maxevents[iConf],
                       gt[iConf],allFromGT[iConf],
                       alignmentDB[iConf], alignmentTAG[iConf],
                       apeDB[iConf], apeTAG[iConf],
                       bowDB[iConf], bowTAG[iConf],
                       vertextype[iConf], tracktype[iConf],
                       applyruncontrol[iConf],
                       ptcut[iConf],input_CMSSW_BASE,AnalysisStep_dir)
            
            aJob.setEOSout(eosdir)
            aJob.createTheCfgFile(theSrcFiles)
            aJob.createTheLSFFile()

            output_file_list1.append("xrdcp root://eoscms//eos/cms"+aJob.getOutputFileName()+" /tmp/$USER/"+opts.taskname+" \n")
            if jobN == 0:
                output_file_list2.append("/tmp/$USER/"+opts.taskname+"/"+aJob.getOutputBaseName()+".root ")
            output_file_list2.append("/tmp/$USER/"+opts.taskname+"/"+os.path.split(aJob.getOutputFileName())[1]+" ")    
   
            if opts.submit:
                aJob.submit()
                batchJobIds.append(ajob.getBatchjobId())
            del aJob

        if opts.submit:
            print("********************************************************")
            for theBatchJobId in batchJobIds:
                print("theBatchJobId is: ",theBatchJobId)

        fout.write("#!/bin/bash \n")
        fout.write("MAIL = $USER@mail.cern.ch \n")
        fout.write("OUT_DIR = "+eosdir+ "\n")
        fout.write("echo $HOST | mail -s \"Harvesting job started\" $USER@mail.cern.ch \n")
        fout.write("cd "+os.path.join(input_CMSSW_BASE,"src")+"\n")
        fout.write("eval `scram r -sh` \n")
        fout.write("mkdir -p /tmp/$USER/"+opts.taskname+" \n")
        fout.writelines(output_file_list1)
        fout.writelines(output_file_list2)
        fout.write("\n")
        fout.write("echo \"xrdcp -f $FILE root://eoscms//eos/cms$OUT_DIR\" \n")
        fout.write("xrdcp -f root://eoscms//eos/cms$FILE $OUT_DIR \n")
        fout.write("echo \"Harvesting for complete; please find output at $OUT_DIR \" | mail -s \"Harvesting for" +opts.taskname +" compled\" $MAIL \n")

        os.system("chmod u+x "+hadd_script_file)

        conditions = '"' + " && ".join(["ended(" + jobId + ")" for jobId in batchJobIds]) + '"'
        print(conditions)
        lastJobCommand = "bsub -o harvester"+opts.taskname+".tmp -q 1nh -w "+conditions+" "+hadd_script_file
        print(lastJobCommand)
        if opts.submit:
            lastJobOutput = getCommandOutput(lastJobCommand)
            print(lastJobOutput)

        fout.close()
        del output_file_list1
        
if __name__ == "__main__":        
    main()
