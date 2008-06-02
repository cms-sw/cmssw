import FWCore.ParameterSet.Config as cms
import relval_common_module as common
import relval_generation_module as generator

import os
import sys 

#import relval_parameters_module as parameters
#Try to eliminate the problem of the _commonsequence without the import
execfile("relval_parameters_module.py")

# This just simplifies the use of the common.logger
mod_id="["+os.path.basename(sys._getframe().f_code.co_filename)[:-3]+"]"


#--------------------------------------------
# Here the functions to add to the process the various steps are defined:
# Build a dict whose keys are the step names and whose values are the functions that 
# add to the process schedule the steps.
def gen(process,name,step,evt_type,energy,evtnumber):
    '''
    Builds the source for the process
    '''
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"
    
    process=generator.generate(process,step,evt_type,energy,evtnumber)
    
    if ( name == 'pgen'):
        process.generation_step = cms.Path(getattr(process,name))
    else:
        process.generation_step = cms.Path(getattr(process,'pgen')*getattr(process,name))
        
    if user_schedule=='':
        process.schedule.append(process.generation_step)
        
    common.log ('%s adding step ...'%func_id)
    return process
    
def sim(process,name,genfilt):
    '''
    Enrich the schedule with simulation
    '''
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"

    if ( genfilt==''):
        process.simulation_step = cms.Path(getattr(process,name))
    else:
        process.simulation_step = cms.Path(getattr(process,genfilt)*getattr(process,name))
        
    if user_schedule=='':
        process.schedule.append(process.simulation_step)  
    
    common.log ('%s adding step ...'%func_id)
    return process
   
def digi(process,name,genfilt):
    '''
    Enrich the schedule with digitisation
    '''
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"

    if ( genfilt==''):
        process.digitisation_step=cms.Path(getattr(process,name))
    else:
        process.digitisation_step = cms.Path(getattr(process,genfilt)*getattr(process,name))

    if user_schedule=='':
        process.schedule.append(process.digitisation_step)
    
    common.log ('%s adding step ...'%func_id)
    
    return process            
       
def reco(process,name,genfilt):
    '''
    Enrich the schedule with reconstruction
    '''
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"
    
#    process.reconstruction_step=cms.Path(process.reconstruction_woConv)
    if ( genfilt==''):
        process.reconstruction_step=cms.Path(getattr(process,name))
    else:
        process.reconstruction_step = cms.Path(getattr(process,genfilt)*getattr(process,name))

    if user_schedule=='':
        process.schedule.append(process.reconstruction_step)     

    common.log ('%s adding step ...'%func_id)
    
    return process            

def l1_trigger(process,name,genfilt):
    '''
    Enrich the schedule with L1 trigger
    '''     
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"
    
    if ( genfilt==''):
        process.L1_Emulation = cms.Path(getattr(process,name))
    else:
        process.L1_Emulation = cms.Path(getattr(process,genfilt)*getattr(process,name))

    if user_schedule=='':
        process.schedule.append(process.L1_Emulation)

    common.log ('%s adding step ...'%func_id)
    
    return process            
    
def postreco_gen(process,name,genfilt):
    '''
    Enrich the schedule with post-reconstruction generator
    '''
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"
    
    if ( genfilt==''):
        process.postreco_generator_step=cms.Path(process.getattr(process,name))
    else:
        process.postreco_generator_step = cms.Path(getattr(process,genfilt)*getattr(process,name))

    if user_schedule=='':
        process.schedule.append(process.postreco_generator_step)     

    common.log ('%s adding step ...'%func_id)
    
    return process            

def ana(process,name,genfilt):
    '''
    Enrich the schedule with analysis
    '''     
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"
    
    if ( genfilt==''):
        process.analysis_step=cms.Path(getattr(process,name))
    else:
        process.analysis_step = cms.Path(getattr(process,genfilt)*getattr(process,name))

    if user_schedule=='':
        process.schedule.append(process.analysis_step)

    common.log ('%s adding step ...'%func_id)
    
    return process            

def digi2raw(process,name,genfilt):
    '''
    Enrich the schedule with raw2digistep
    '''     
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"
    
    if ( genfilt==''):
        process.digi2raw_step=cms.Path(getattr(process,name))
    else:
        process.digi2raw_step = cms.Path(getattr(process,genfilt)*getattr(process,name))

    if user_schedule=='':
        process.schedule.append(process.digi2raw_step)
    
    common.log ('%s adding step ...'%func_id)
    
    return process

def raw2digi(process,name,genfilt):
    '''
    Enrich the schedule with raw2digistep
    '''     
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"
    
    if ( genfilt==''):
        process.raw2digi_step=cms.Path(getattr(process,name))
    else:
        process.raw2digi_step = cms.Path(getattr(process,genfilt)*getattr(process,name))

    if user_schedule=='':
        process.schedule.append(process.raw2digi_step)
    
    common.log ('%s adding step ...'%func_id)
    
    return process

def validation(process,name,genfilt):
    '''
    Enrich the schedule with validation
    '''
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"
    
    if ( genfilt==''):
        process.validation_step=cms.Path(getattr(process,name))
    else:
        process.validation_step = cms.Path(getattr(process,genfilt)*getattr(process,name))

    if user_schedule=='':
        process.schedule.append(process.validation_step)
    
    common.log ('%s adding step ...'%func_id)
    return process            

def hlt(process,name,genfilt):
    '''
    Enrich the schedule with hlt
    '''
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"

    common.log ('%s adding hlt paths ...'%func_id)

    thePath= os.environ['CMSSW_SEARCH_PATH']
    theFileName = 'HLTrigger/Configuration/data/HLT_2E30.cff'
#    theFileName = 'HLTrigger/Configuration/data/HLT_1E32.cff'
    pathList=thePath.split(':')
    fullName=''
    for path in pathList :
        fullName=path+'/'+theFileName
        if os.path.exists(fullName) :
           break
    print 'Found: '+fullName
    theFile=file(fullName)

    sortedPaths = []
    endPaths= []

    # parse the file for path definitions
    for line in theFile.read().splitlines():
        if line.startswith("path"):
            sortedPaths.append(line.split()[1]) #that's the pathname
            print 'appending ' + line.split()[1]
        if line.startswith("endpath"):
            endPaths.append(line.split()[1]) #that's the pathname
            
    theFile.close() 

    if user_schedule=='':
        for path in sortedPaths:
            if path.startswith("HLT") or path.startswith("CandHLT") or path.startswith("AlCa"):
                process.schedule.append(getattr(process,path)) 
                common.log ('%s path added  ...'%path)
                
    if ( len(endPaths)>1 ):
        print 'Hum, pyrelval can not parse multiple hlt endpaths. Ask for help\n'
        sys.exit(0)

    if ( len(endPaths)==1):
         process.hltEndPath=getattr(process,endPaths[0])

    if ( genfilt!=''):
        # unfortunately not every HLT paths uses the HLTBeginSequence. Only trust what you tested yourself... :-/
        # process.HLTBeginSequence._seq = cms.Sequence(process.ProductionFilterSequence._seq*process.HLTBeginSequence._seq) 
        for path in sortedPaths:
            if path.startswith("HLT") or path.startswith("CandHLT") or path.startswith("AlCa"):
                getattr(process,path)._seq = process.ProductionFilterSequence._seq*getattr(process,path)._seq
#    for p  in process.paths_().itervalues():
#        pname=p.label()
#        if ( pname[0:3]=='HLT' or pname[0:7]=='CandHLT' ):
#            process.schedule.append(getattr(process,pname))
#            common.log ('%s path added  ...'%pname)

    return process

def fastsim(process,name,genfilt):
    '''
    Enrich the schedule with fastsim
    '''
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"
    
    if ( genfilt==''):
        process.fastsim_step=cms.Path(getattr(process,name))
    else:
        process.fastsim_step = cms.Path(getattr(process,genfilt)*getattr(process,name))

    if user_schedule=='':
        process.schedule.append(process.fastsim_step)
    
    common.log ('%s adding step ...'%func_id)
    
    return process            

def alca(process,name,genfilt):
    '''
    Enrich the schedule with alcareco
    '''
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"

# if paths not specified, just take everything and hope...
    if user_schedule=='':
        if ( name ==''):
            for p  in process.paths_().itervalues():
                pname=p.label()
                if ( pname[0:12]=='pathALCARECO'):
                    process.schedule.append(getattr(process,pname))
                    common.log ('%s path added  ...'%pname)
        else:
            paths=name.split('+')
            for p in paths:
                process.schedule.append(getattr(process,'pathALCARECO'+p))
                common.log ('%s path added  ...'%p)
            
    return process

def postreco(process,name,genfilt):
    '''
    Enrich the schedule with postreco
    '''
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"
    
    if ( genfilt==''):
        process.postreco_step=cms.Path(getattr(process,name))
    else:
        process.postreco_step = cms.Path(getattr(process,genfilt)*getattr(process,name))
    if user_schedule=='':
        process.schedule.append(process.postreco_step)
    
    common.log ('%s adding step ...'%func_id)
    
    return process            
