import FWCore.ParameterSet.Config as cms
import relval_common_module as common

import os
import sys 

#import relval_parameters_module as parameters
#Try to eliminate the problem of the _commonsequence without the import
execfile("relval_parameters_module.py")

# This just simplifies the use of the common.logger
mod_id="["+os.path.basename(sys._getframe().f_code.co_filename)[:-3]+"]"

# At top level and not in a function. To be fixed
# The priority with wich the generators module is seeked for..
generator_module_name="relval_generation_module.py"
pyrelval_location=os.environ["CMSSW_BASE"]+"/src/Configuration/PyReleaseValidation/python/"+generator_module_name
pyrelval_release_location=os.environ["CMSSW_RELEASE_BASE"]+"/src/Configuration/PyReleaseValidation/python/"+generator_module_name

locations=(pyrelval_location,
           pyrelval_release_location)
        
mod_location=""
for location in locations:
    if os.path.exists(location):
        mod_location=location

print 'mod_location %s' % mod_location
execfile(mod_location)
print generate

#--------------------------------------------
# Here the functions to add to the process the various steps are defined:
# Build a dict whose keys are the step names and whose values are the functions that 
# add to the process schedule the steps.
def gen(process,name,step,evt_type,energy,evtnumber):
    '''
    Builds the source for the process
    '''
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"
    

    process.source=generate(step,evt_type,energy,evtnumber)
    process.generation_step = cms.Path(getattr(process,name))
    if not user_schedule:
        process.schedule.append(process.generation_step)
        
    common.log ('%s adding step ...'%func_id)
    return process
    
def sim(process,name):
    '''
    Enrich the schedule with simulation
    '''
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"

    process.simulation_step = cms.Path(getattr(process,name))
    if not user_schedule:
        process.schedule.append(process.simulation_step)  
    
    common.log ('%s adding step ...'%func_id)
    return process
   
def digi(process,name):
    '''
    Enrich the schedule with digitisation
    '''
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"
    
    process.digitisation_step=cms.Path(getattr(process,name))
    if not user_schedule:
        process.schedule.append(process.digitisation_step)
    
    common.log ('%s adding step ...'%func_id)
    
    return process            
       
def reco(process,name):
    '''
    Enrich the schedule with reconstruction
    '''
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"
    
#    process.reconstruction_step=cms.Path(process.reconstruction_woConv)
    process.reconstruction_step=cms.Path(getattr(process,name))
    if not user_schedule:
        process.schedule.append(process.reconstruction_step)     

    common.log ('%s adding step ...'%func_id)
    
    return process            

def l1_trigger(process,name):
    '''
    Enrich the schedule with L1 trigger
    '''     
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"
    
    process.L1_Emulation = cms.Path(getattr(process,name))
    if not user_schedule:
        process.schedule.append(process.L1_Emulation)

    common.log ('%s adding step ...'%func_id)
    
    return process            
    
def postreco_gen(process,name):
    '''
    Enrich the schedule with post-reconstruction generator
    '''
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"
    
    process.postreco_generator_step=cms.Path(process.getattr(process,name))
    if not user_schedule:
        process.schedule.append(process.postreco_generator_step)     

    common.log ('%s adding step ...'%func_id)
    
    return process            

def ana(process,name):
    '''
    Enrich the schedule with analysis
    '''     
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"
    
    process.analysis_step=cms.Path(getattr(process,name))
    if not user_schedule:
        process.schedule.append(process.analysis_step)

    common.log ('%s adding step ...'%func_id)
    
    return process            

def digi2raw(process,name):
    '''
    Enrich the schedule with raw2digistep
    '''     
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"
    
    process.digi2raw_step=cms.Path(getattr(process,name))
    if not user_schedule:
        process.schedule.append(process.digi2raw_step)
    
    common.log ('%s adding step ...'%func_id)
    
    return process

def raw2digi(process,name):
    '''
    Enrich the schedule with raw2digistep
    '''     
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"
    
    process.raw2digi_step=cms.Path(getattr(process,name))
    if not user_schedule:
        process.schedule.append(process.raw2digi_step)
    
    common.log ('%s adding step ...'%func_id)
    
    return process

def validation(process,name):
    '''
    Enrich the schedule with validation
    '''
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"
    
    process.validation_step=cms.Path(getattr(process,name))
    if not user_schedule:
        process.schedule.append(process.validation_step)
    
    common.log ('%s adding step ...'%func_id)
    return process            

def hlt(process,name):
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

    for path in sortedPaths:
        if path.startswith("HLT") or path.startswith("CandHLT"):
            process.schedule.append(getattr(process,path)) 
            common.log ('%s path added  ...'%path)

    if ( len(endPaths)>1 ):
        print 'Hum, pyrelval can not parse multiple hlt endpaths. Ask for help\n'
        sys.exit(0)

    if ( len(endPaths)==1):
         process.hltEndPath=getattr(process,endPaths[0])

#    for p  in process.paths_().itervalues():
#        pname=p.label()
#        if ( pname[0:3]=='HLT' or pname[0:7]=='CandHLT' ):
#            process.schedule.append(getattr(process,pname))
#            common.log ('%s path added  ...'%pname)

    return process

def fastsim(process,name):
    '''
    Enrich the schedule with fastsim
    '''
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"
    
    process.offlinedqm_step=cms.Path(getattr(process,name))
    if not user_schedule:
        process.schedule.append(process.fastsim)
    
    common.log ('%s adding step ...'%func_id)
    
    return process            

def alca(process,name):
    '''
    Enrich the schedule with alcareco
    '''
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"

    for p  in process.paths_().itervalues():
        pname=p.label()
        if ( pname[0:12]=='pathALCARECO'):
            process.schedule.append(getattr(process,pname))
            common.log ('%s path added  ...'%pname)
            
    return process

def postreco(process,name):
    '''
    Enrich the schedule with postreco
    '''
    func_id=mod_id+"["+sys._getframe().f_code.co_name+"]"
    
    process.postreco_step=cms.Path(getattr(process,name))
    if not user_schedule:
        process.schedule.append(process.postreco_step)
    
    common.log ('%s adding step ...'%func_id)
    
    return process            
