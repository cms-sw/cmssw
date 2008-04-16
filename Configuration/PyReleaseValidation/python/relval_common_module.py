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

def add_includes(process,PU_flag,step_list,conditions,beamspot):
    """Function to add the includes to the process.
    It returns a process enriched with the includes.
    """

    conditionsSP=conditions.split(',')

#these are the defaults..
    incList=['"Configuration/StandardSequences/data/Services.cff"',
             '"Configuration/StandardSequences/data/Geometry.cff"',
             '"Configuration/StandardSequences/data/MagneticField.cff"',
             '"FWCore/MessageService/data/MessageLogger.cfi"',
             '"Configuration/StandardSequences/data/VtxSmeared'+beamspot+'.cff"',
             '"Configuration/StandardSequences/data/Generator.cff"',             
             '"Configuration/StandardSequences/data/'+conditionsSP[0]+'.cff"']             

    if PU_flag:
        incList.append('"Configuration/StandardSequences/data/MixingLowLumiPileUp.cff"')
    else:
        incList.append('"Configuration/StandardSequences/data/MixingNoPileUp.cff"')

    inc_dict={'GEN':'"Configuration/StandardSequences/data/Generator.cff"',
              'SIM':'"Configuration/StandardSequences/data/Simulation.cff"',
              'DIGI':'',
              'RECO':'"Configuration/StandardSequences/data/Reconstruction.cff"',
              'ALCA':'"Configuration/StandardSequences/data/AlCaReco.cff"',
              'L1':'"Configuration/StandardSequences/data/L1Emulator.cff" "Configuration/StandardSequences/data/L1TriggerDefaultMenu.cff"',
              'DIGI2RAW':'"Configuration/StandardSequences/data/DigiToRaw.cff"',
              'RAW2DIGI':'"Configuration/StandardSequences/data/RawToDigi.cff"',
              'ANA':'"Configuration/StandardSequences/data/Analysis.cff"',
              'DQM':'"Configuration/StandardSequences/data/Validation.cff"',
              'FASTSIM':'"Configuration/StandardSequences/data/FastSimulation.cff"',
              'HLT':'"HLTrigger/Configuration/data/HLT_2E30.cff" "HLTrigger/Configuration/data/common/HLTPrescaleReset.cff"',
              'POSTRECO':'"Configuration/StandardSequences/data/PostRecoGenerator.cff"'
              }

# to get the configs to parse, sometimes other steps are needed..
    dep_dict={'DIGI':'SIM','ALCA':'RECO','HLT':'GEN:SIM:DIGI:L1'}
    
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"
    log(func_id+" Entering... ")
        
    # The file FWCore/Framework/test/cmsExceptionsFatalOption.cff:
    fataloptions="FWCore/Framework/test/cmsExceptionsFatalOption.cff" 
    fataloptions_inclobj=include_files(fataloptions)[0]

    process.options=cms.untracked.PSet\
                     (Rethrow=fataloptions_inclobj.Rethrow,
                      wantSummary=cms.untracked.bool(True),
                      makeTriggerResults=cms.untracked.bool(True) ) 
    
                 
    from FWCore.ParameterSet.parseConfig import parseConfigString

#try to protect against including multiple times (may or may not matter to speed)
    sourcedList=['none']
    
    for s in step_list:
        stepSP=s.split(':') 
        step=stepSP[0]

# first look for dependencies and add them to the include list
        if ( dep_dict.has_key(step)):
            depSP=dep_dict[step].split(':')
            for dep in depSP:
                if dep not in sourcedList:
                    sourcedList.append(dep)
                    if inc_dict[dep] != '':
                        incs=inc_dict[dep].split(' ')
                        for inc in incs:
                            incList.append(inc)
        if inc_dict[step] != '':
            incs=inc_dict[step].split(' ')
            for inc in incs:
                incList.append(inc)
        sourcedList.append(step)                    

    stringToInclude=''        
    for incF in incList:
        stringToInclude=stringToInclude+'include '+incF+' ' 

    process.extend(parseConfigString(stringToInclude))
    if ( len(conditionsSP)>1 ):
        process.GlobalTag.globaltag=conditionsSP[1]

    log(func_id+ " Returning process...")
    return process

#-----------------------------------------

def event_input(infile_name,second_name):
    """
    Returns the source for the process.
    """ 
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"
    if ( second_name !='' ):
        pr_source=cms.Source("PoolSource",
                             fileNames = cms.untracked.vstring\
                             ((infile_name)),
                             secondaryFileNames = cms.untracked.vstring\
                             ((second_name)))
    else:    
        pr_source=cms.Source("PoolSource",
                             fileNames = cms.untracked.vstring\
                             ((infile_name)))
        
    log(func_id+" Adding PoolSource source ...")                         
    return pr_source
    
#-----------------------------------------

def event_output(process, outfile_name, step, eventcontent, evt_filter=None):
    """
    Function that enriches the process so to produce an output.
    """ 
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"
    # Event content

    content=include_files("Configuration/EventContent/data/EventContent.cff")[0]
        
    process.extend(content)
    if hasattr(process,'generation_step'):
        process.out_step = cms.OutputModule\
                           ("PoolOutputModule",
                            outputCommands=getattr(content,eventcontent+'EventContent').outputCommands,
                            fileName = cms.untracked.string(outfile_name),
                            dataset = cms.untracked.PSet(dataTier =cms.untracked.string(step)),
                            SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('generation_step'))
                            ) 
    else:
        process.out_step = cms.OutputModule\
                           ("PoolOutputModule",
                            outputCommands=getattr(content,eventcontent+'EventContent').outputCommands,
                            fileName = cms.untracked.string(outfile_name),
                            dataset = cms.untracked.PSet(dataTier =cms.untracked.string(step))
                            ) 
    
    process.outpath = cms.EndPath(process.out_step)
    
    log(func_id+" Adding PoolOutputModule ...") 
    
    return process 

def raw_output(process, outfile_name, evt_filter=None):
    """
    Function that enriches the process so to produce an output.
    """ 
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"

    process.out_step_raw = cms.OutputModule\
                           ("PoolOutputModule",
                            outputCommands=process.RAWEventContent.outputCommands,
                            fileName = cms.untracked.string(outfile_name),
                            dataset = cms.untracked.PSet(dataTier =cms.untracked.string('RAW')),
                            SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('generation_step'))
                           ) 
    
    process.outpath_raw = cms.EndPath(process.out_step_raw)
    
    log(func_id+" Adding PoolOutputModule for raw file ...") 
    
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
              (version=cms.untracked.string("$Revision: 1.11 $"),
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
                
