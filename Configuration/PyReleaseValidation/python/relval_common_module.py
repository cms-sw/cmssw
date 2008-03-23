#####################################################
#                                                   # 
#               relval_common_module                #
#                                                   #
#  Module that collects the functions to carry out  #
#  to the operations necessary for release          #
#  validation. It includes the building of the      # 
#  message logger and the IO.                       #
#                                                   #
#####################################################

__author__  = "Danilo Piparo"


import FWCore.ParameterSet.Config as cms

import pickle 
import os # To check the existance of pkl objects files
import sys # to get current funcname
import time

# This just simplifies the use of the logger
mod_id="["+os.path.basename(sys._getframe().f_code.co_filename)[:-3]+"]"

#------------------------

def include_files(includes_set):
    """
    It takes a string or a list of strings and returns a list of 
    FWCore.ParameterSet.parseConfig._ConfigReturn objects.
    In the package directory it creates ASCII files in which the objects are coded. If 
    the files exist already it symply loads them.
    """
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"
        
    packagedir="./"
    #Trasform the includes_set in a list
    if not isinstance(includes_set,list):
        includes_set=[includes_set]
    
    object_list=[]    
    for cf_file_name in includes_set:
        pkl_file_name=packagedir+os.path.basename(cf_file_name)[:-4]+".pkl"
        
        cf_file_fullpath=""
        # Check the paths of the cffs
        for path in os.environ["CMSSW_SEARCH_PATH"].split(":"):
            cf_file_fullpath=path+"/"+cf_file_name
            if os.path.exists(cf_file_fullpath):
                break
        
        pkl_file_exists=os.path.exists(pkl_file_name)               
        # Check the dates of teh cff and the corresponding pickle
        cff_age=0
        pkl_age=0
        if pkl_file_exists:
            cff_age=os.path.getctime(cf_file_fullpath)
            pkl_age=os.path.getctime(pkl_file_name)
            if cff_age>pkl_age:
                log(func_id+" Pickle object older than file ...")
        
       
        if not pkl_file_exists or cff_age>pkl_age:
          obj=cms.include(cf_file_name)
          file=open(pkl_file_name,"w")
          pickle.dump(obj,file)   
          file.close()
          log(func_id+" Pickle object for "+cf_file_fullpath+" dumped as "+pkl_file_name+"...")
        # load the pkl files.                       
        file=open(pkl_file_name,"r")
        object_list.append(pickle.load(file))
        file.close()
        log(func_id+" Pickle object for "+cf_file_fullpath+" loaded ...")
    
    return object_list
    
#------------------------

def add_includes(process,PU_flag,step_list):
    """Function to add the includes to the process.
    It returns a process enriched with the includes.
    """

    inc_dict={'GEN':'',
              'SIM':'',
              'DIGI':'',
              'RECO':'',
              'L1':'',
              'DIGI2RAW':'',
              'ANA':'',
              'DQM':'',
              'FASTSIM':'Configuration/StandardSequences/data/FastSimulation.cff',
              'HLT':'Configuration/StandardSequences/data/HLT.cff'}
    
    
    
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"
    log(func_id+" Entering... ")
        
    for included_fragment in include_files(["Configuration/StandardSequences/data/Services.cff"]): 
        process.extend(included_fragment)
    

    # The file FWCore/Framework/test/cmsExceptionsFatalOption.cff:
    fataloptions="FWCore/Framework/test/cmsExceptionsFatalOption.cff" 
    fataloptions_inclobj=include_files(fataloptions)[0]
    
    process.options=cms.untracked.PSet\
                    (Rethrow=fataloptions_inclobj.Rethrow,
                     wantSummary=cms.untracked.bool(True),
                     makeTriggerResults=cms.untracked.bool(True) ) 
                 
    process.extend(include_files("FWCore/MessageService/data/MessageLogger.cfi")[0])                  

    if PU_flag:
        process.extend(include_files("Configuration/PyReleaseValidation/data/incl_summary_PU.cff")[0])   
    else:
        process.extend(include_files("Configuration/PyReleaseValidation/data/incl_summary.cff")[0])   

    for s in step_list:
        stepSP=s.split(':') 
        step=stepSP[0]
        if inc_dict[step] != '':
           included_fragment=inc_dict[step]
           process.extend(include_files(included_fragment)[0])


    
    log(func_id+ " Returning process...")
    return process

#-----------------------------------------

def event_input(infile_name):
    """
    Returns the source for the process.
    """ 
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"
    pr_source=cms.Source("PoolSource",
                         fileNames = cms.untracked.vstring\
                                     ((infile_name)))
    log(func_id+" Adding PoolSource source ...")                         
    return pr_source
    
#-----------------------------------------

def event_output(process, outfile_name, step, evt_filter=None):
    """
    Function that enriches the process so to produce an output.
    """ 
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"
    # Event content
    content=include_files("Configuration/EventContent/data/EventContent.cff")[0]
    process.extend(content)
    process.out_step = cms.OutputModule\
                   ("PoolOutputModule",
                    outputCommands=content.FEVTSIMDIGIEventContent.outputCommands,
                    fileName = cms.untracked.string(outfile_name),
                    dataset = cms.untracked.PSet(dataTier =cms.untracked.string(step))
                   ) 
    
    process.outpath = cms.EndPath(process.out_step)
    
    log(func_id+" Adding PoolOutputModule ...") 
    
    return process 
 
#--------------------------------------------
    
def build_profiler_service(evts_cuts):
    """
    A profiler service by Vincenzo Innocente.
    """
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"
    
    firstevent=int(evts_cuts.split("_")[0])
    lastevent=int(evts_cuts.split("_")[1])
    prof_service=cms.Service("ProfilerService",
                             firstEvent=cms.untracked.int32(firstevent),
                             lastEvent=cms.untracked.int32(lastevent),
                             paths=cms.untracked.vstring("FullEvent")                        
                            )
                            
    log(func_id+" Returning Service...")
                                                        
    return prof_service
    
#--------------------------------------------------- 

def build_fpe_service(options="1110"):
    """
    A service for trapping floating point exceptions
    """
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"
    
    fpe_service=cms.Service("EnableFloatingPointExceptions",
                            enableDivByZeroEx=cms.untracked.bool(bool(options[0])),
                            enableInvalidEx=cms.untracked.bool(bool(options[1])),
                            enableOverflowEx=cms.untracked.bool(bool(options[2])),
                            enableUnderflowEx=cms.untracked.bool(bool(options[3]))
                           )  
    
    log(func_id+" Returning Service...")
                             
    return fpe_service
    
#---------------------------------------------------

def build_production_info(evt_type, energy, evtnumber):
    """
    Add useful info for the production.
    """
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"
    
    prod_info=cms.untracked.PSet\
              (version=cms.untracked.string("$Revision: 1.40 $"),
               name=cms.untracked.string("PyReleaseValidation"),
               annotation=cms.untracked.string(evt_type+" energy:"+str(energy)+" nevts:"+str(evtnumber))
              )
    

    log(func_id+" Adding Production info ...")              
              
    return prod_info 

#--------------------------------------------

def log (message):
    """
    An oversimplified logger. This is designed for debugging the PyReleaseValidation
    """
    hour=time.asctime().split(" ")[3]
    #if parameters.dbg_flag:
    if True:    
        print "["+hour+"]"+message
                
