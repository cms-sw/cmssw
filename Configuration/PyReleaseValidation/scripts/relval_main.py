#!/usr/bin/env python
###################################################
#                                                 #
#                 relval_main                     #
#                                                 #              
#  Release validation main file. It initialises   #
#  the process and uses the informations kept in  #
#  relval_parameters_module to build the object.  #
#                                                 #
###################################################

__author__="Danilo Piparo"

# Let Python find the parameters module created locally in the current directory.
# As long as the Python code cannot have any command line arguments since this could lead
# to conflicts with cmsRun this is a way to input 
import sys
import pickle
import os

# Modules to include

import FWCore.ParameterSet.Config as cms

#import relval_parameters_module as parameters
#Try to eliminate the problem of the _commonsequence without the import

execfile("relval_parameters_module.py")
import Configuration.PyReleaseValidation.relval_common_module as common
import Configuration.PyReleaseValidation.relval_steps_module as steps 

sys.path.append(".") # necessary to find the relval_parameters_module created by CMSdriver

# parse the string containing the steps and make a list out of it
if step=='ALL':
    step='GEN,SIM,DIGI,L1,DIGI2RAW,RECO,DQM,POSTRECO'
step_list=step.split(',') # we split when we find a ','

# a dict whose keys are the steps and the values are functions that modify the process
# in order to add the actual step..
step_dict={'GEN':steps.gen,
           'SIM':steps.sim,
           'DIGI':steps.digi,
           'RECO':steps.reco,
           'ALCA':steps.alca,
           'L1':steps.l1_trigger,
           'DIGI2RAW':steps.digi2raw,
           'RAW2DIGI':steps.raw2digi,
           'ANA':steps.ana,
           'DQM':steps.validation,
           'FASTSIM':steps.fastsim,
           'HLT':steps.hlt,
           'POSTRECO':steps.postreco}

#these are junk.. what should these be?
dataTier_dict={'GEN':'GEN',
               'SIM':'SIM',
               'DIGI':'DIGI',
               'RECO':'RECO',
               'ALCA':'RECO',
               'L1':'L1',
               'DIGI2RAW':'RAW',
               'RAW2DIGI':'DIGI',
               'ANA':'RECO',
               'DQM':'RECO',
               'FASTSIM':'RECO',
               'HLT':'GEN-SIM-RAW',
               'POSTRECO':'RECO'}

pathName_dict={'GEN':'pgen',
               'SIM':'psim',
               'DIGI':'pdigi',
               'RECO':'reconstruction',
               'ALCA':'',
               'L1':'L1Emulator',
               'DIGI2RAW':'DigiToRaw',
               'RAW2DIGI':'RawToDigi',
               'ANA':'analysis',
               'DQM':'validation',
               'FASTSIM':'fastsim',
               'HLT':'hlt',
               'POSTRECO':'postreco_generator'}

#---------------------------------------------------

# Here the process is built according to the settings in
# the relval_parameters_module. All the objects built have in 
# common the features described in the relval_common_module.

print "\nPython RelVal"

#HACK - if config contains hlt then process must be HLT.
#Likely to cause problems later - Please complain to the HLT if so.. 
for s in step_list:
    stepSP=s.split(':') 
    step=stepSP[0]
    if ( step=='HLT'):
        process_name='HLT'
 
process = cms.Process (process_name)
         
process.schedule=cms.Schedule()

# Enrich the process with the features described in the relval_includes_module.
process=common.add_includes(process,PU_flag,step_list,conditions,beamspot)

# Add the fpe service if needed
if fpe_service_flag:
    process.add_(common.build_fpe_service()) 

# Add the Profiler Service if needed:
if profiler_service_cuts!="":
    process.add_(common.build_profiler_service(profiler_service_cuts))

# Set the number of events with a top level PSet
process.maxEvents=cms.untracked.PSet(input=cms.untracked.int32(evtnumber))

# Add the ReleaseValidation PSet
totnumevts,evtsperjob=releasevalidation
dsetname="RelVal"+ext_process_name
if dump_pickle!='':
    dsetname=dump_pickle
process.ReleaseValidation=cms.untracked.PSet(totalNumberOfEvents=cms.untracked.int32(totnumevts),
                                             eventsPerJob=cms.untracked.int32(evtsperjob),
                                             primaryDatasetName=cms.untracked.string("RelVal"+dsetname.replace('.pkl','')))


# Add metadata for production - only if not already present
if not hasattr(process,"configurationMetadata"):
    process.configurationMetadata=common.build_production_info(evt_type, energy, evtnumber)
 
"""
Here we choose to make the process work only for one of the four steps 
(GEN,SIM DIGI RECO) or for the whole chain (ALL)
"""

isFirst=0
genfilt=''
for s in step_list:
    stepSP=s.split(':') 
    step=stepSP[0]
    pathname=''
    if ( len(stepSP)>1):
        pathname=stepSP[1]
    if ( pathname == '' ):
       pathname=pathName_dict[step]
    if ( isFirst==0 ):   
        if step in ('GEN'):
            if ( pathname != pathName_dict[step]):
                genfilt=pathname
            process=steps.gen(process,pathname,step,evt_type,energy,evtnumber)           
        else:
            process.source = common.event_input(infile_name,insecondfile_name) 
            process=step_dict[step](process,pathname,genfilt)                      
        isFirst=1
    else:    
        process=step_dict[step](process,pathname,genfilt)                      

#look for an hlt endpath
if user_schedule=='':
    if hasattr(process,'hltEndPath'):
        process.schedule.append(process.hltEndPath)

#override the data tier (likely the usual..)
    

# Add the output on a root file if requested
if output_flag:
    outputDT=dataTier_dict[step]
    if ( dataTier!=''):
        outputDT=dataTier
    if ( not (eventcontent == 'none') ):    
        process = common.event_output\
                  (process, outfile_name, outputDT,eventcontent,filtername, conditions)
        if user_schedule=='':
            process.schedule.append(process.outpath)  
    if ( rawfile_name!='' ):
        print 'Add raw file' + rawfile_name
        process = common.raw_output\
                  (process, rawfile_name)
        if user_schedule=='':
            process.schedule.append(process.outpath_raw)  

# add the user schedule here..
if user_schedule!='':
    print 'hi there'
    for k in user_schedule.split(','):
        print k
        process.schedule.append(getattr(process,k))

# look for alca reco
    nALCA=0
    for k in process.schedule:
        if ( k.label()[0:12]=='pathALCARECO'):
            nALCA=nALCA+1
            if nALCA==1:
                content=common.include_files("Configuration/EventContent/data/AlCaRecoOutput.cff")[0]
                process.extend(content)
            poUsing='Out'+k.label()[4:len(k.label())]
            filterName=k.label()[12:len(k.label())]
            rootName='file:'+filterName+'.root'
            modName='pool'+k.label()[4:len(k.label())]
            pathName='outPath'+k.label()[4:len(k.label())]
            if ( hasattr(process,poUsing)):
                if not oneoutput:
                    poolOutT = cms.OutputModule("PoolOutputModule",getattr(process,poUsing),\
                                                dataset = cms.untracked.PSet(filterName = cms.untracked.string(filterName),\
                                                                             dataTier = cms.untracked.string('ALCARECO')),\
                                                fileName = cms.untracked.string(rootName)
                                                )
                    setattr(process,modName,poolOutT)
                    setattr(process,pathName,cms.EndPath(getattr(process,modName)))
#                    if user_schedule=='':
                    process.schedule.append(getattr(process,pathName))
                else:
                    # add the alca outputs into the main output file 
                    alcaOutput=getattr(process,poUsing).outputCommands
                    for a in alcaOutput:
                        if ( "drop" not in a ):
                            process.out_step.outputCommands.append(a)
    if ( nALCA>0):        
        print 'Number of AlCaReco output streams added: '+str(nALCA)

# generic config corrections go here
process.GlobalTag.DBParameters.connectionTimeOut=60



# Add a last customisation of the process as specified in the file.
if customisation_file!='':
    file=__import__(customisation_file[:-3])
    process=file.customise(process)
    if process == None:
        raise ValueError("Customise file returns no process. Please add a 'return process'.")
        
    
# print to screen the config file in the old language
print dump_cfg
if dump_cfg!='':
    cfg=open(dump_cfg,'w') 
    cfg.write(process.dumpConfig()) #this used to be dumpPython.. No idea why?
    cfg.close()
    print 'the following exception is normal! You can ignore it'
    sys.exit() # no need to launch the FW

# print to screen the config file in the python language
if dump_python!='':
    pycfg=open(dump_python,'w') 
    pycfg.write('import FWCore.ParameterSet.Config as cms \n')
    pycfg.write(process.dumpPython())
    pycfg.close()
    print 'the following exception is normal! You can ignore it'
    sys.exit() # no need to launch the FW


# dump a pickle object of the process on disk:
if dump_pickle!='':
   print "Dumping process on disk as a pickle object..."
   pickle_file=file(dump_pickle,"w")
   pickle.dump(process,pickle_file)
   pickle_file.close()
   sys.exit() # no need to launch the FW
       
# A sober separator between the python program and CMSSW    
print "And now The Framework -----------------------------"
sys.stdout.flush() 

process.dumpPython()
