#! /usr/bin/env python

# A Pyrelval Wrapper

from __future__ import print_function
import optparse
import sys
import os
import re
import Configuration.Applications
from Configuration.Applications.ConfigBuilder import ConfigBuilder, defaultOptions
import traceback
from functools import reduce

def checkModifier(era):
    from FWCore.ParameterSet.Config import Modifier, ModifierChain
    return isinstance( era, Modifier ) or isinstance( era, ModifierChain )

def checkOptions():
    return
    
def adaptOptions():
    return

def OptionsFromCommand(command):
    items=command.split()
    if items[0] != 'cmsDriver.py':
        return None
    items.append('--evt_type')
    items.append(items[1])
    options=OptionsFromItems(items[2:])
    options.arguments = command
    return options

def OptionsFromCommandLine():
    import sys
    options=OptionsFromItems(sys.argv[1:])
    # memorize the command line arguments 
    options.arguments = reduce(lambda x, y: x+' '+y, sys.argv[1:])
    return options

def OptionsFromItems(items):
    import sys
    from Configuration.Applications.Options import parser,threeValued
    #three valued options
    for (index,item) in enumerate(items):
        for (opt,value) in threeValued:
            if (str(item) in opt) and (index==len(items)-1 or items[index+1].startswith('-')):
                items.insert(index+1,value)
                
    (options,args) = parser.parse_args(items)

    if not options.conditions or options.conditions=="help":
        from Configuration.AlCa import autoCond
        possible=""
        for k in autoCond.autoCond:
            possible+="\nauto:"+k+" -> "+str(autoCond.autoCond[k])
        raise Exception("the --conditions option is mandatory. Possibilities are: "+possible)


    #################################
    # Check parameters for validity #
    #################################

    # check in case of ALCAOUTPUT case for alca splitting
    if options.triggerResultsProcess == None and "ALCAOUTPUT" in options.step:
        print("ERROR: If ALCA splitting is requested, the name of the process in which the alca producers ran needs to be specified. E.g. via --triggerResultsProcess RECO")
        sys.exit(1)
            
    if not options.evt_type:            
        options.evt_type=sys.argv[1]

    #now adjust the given parameters before passing it to the ConfigBuilder

    # concurrency options
    nStreams = options.nStreams if options.nStreams != '0' else options.nThreads
    if options.nConcurrentLumis == '0':
        options.nConcurrentLumis = '1' if nStreams == '1' else '2'
    if options.nConcurrentIOVs == '0':
        options.nConcurrentIOVs = options.nConcurrentLumis

    #trail a "/" to dirin and dirout
    if options.dirin!='' and (not options.dirin.endswith('/')):    options.dirin+='/'
    if options.dirout!='' and (not options.dirout.endswith('/')):  options.dirout+='/'

    # Build the IO files if necessary.
    # The default form of the files is:
    # <type>_<energy>_<step>.root
    prec_step = {"NONE":"",
                 "FILTER":"",
                 "ALL":"",
                 "LHE":"",
                 "GEN":"",
                 "reGEN":"",
                 "SIM":"GEN",
                 "reSIM":"SIM",
                 "DIGI":"SIM",
                 "reDIGI":"DIGI",
                 "L1REPACK":"RAW",
                 "HLT":"RAW",
                 "RECO":"DIGI",
                 "ALCA":"RECO",
                 "ANA":"RECO",
                 "SKIM":"RECO",
                 "DIGI2RAW":"DIGI",
                 "RAW2DIGI":"DIGI2RAW",
                 "RAW2RECO":"DIGI2RAW",
                 "DATAMIX":"DIGI",
                 "DIGI2RAW":"DATAMIX",
                 "HARVESTING":"RECO",
                 "ALCAHARVEST":"RECO",
                 "PAT":"RECO",
                 "NANO":"PAT",
                 "PATGEN":"GEN"}

    trimmedEvtType=options.evt_type.split('/')[-1]

    #get the list of steps, without their options
    options.trimmedStep=[]
    for s in options.step.split(','):
        step=s.split(':')[0]
        options.trimmedStep.append(step)
    first_step=options.trimmedStep[0]

    #replace step aliases
    # this does not affect options.trimmedStep which still contains 'NONE'
    stepsAliases={
        'NONE':'',
        'ALL':'GEN,SIM,DIGI,L1,DIGI2RAW,HLT:GRun,RAW2DIGI,RECO,POSTRECO,VALIDATION,DQM',
        'DATA_CHAIN':'RAW2DIGI,RECO,POSTRECO,DQM'
        }
    if options.step in stepsAliases:
        options.step=stepsAliases[options.step]

    options.step = options.step.replace("SIM_CHAIN","GEN,SIM,DIGI,L1,DIGI2RAW")

    # add on the end of job sequence...
    addEndJob = True
    if ("FASTSIM" in options.step and not "VALIDATION" in options.step) or "HARVESTING" in options.step or "ALCAHARVEST" in options.step or "ALCAOUTPUT" in options.step or options.step == "": 
        addEndJob = False
    if ("SKIM" in options.step and not "RECO" in options.step):
        addEndJob = False
    if ("ENDJOB" in options.step):
        addEndJob = False
    if ('DQMIO' in options.datatier):
        addEndJob = False
    if addEndJob:    
        options.step=options.step+',ENDJOB'


    #determine the type of file on input
    if options.filetype==defaultOptions.filetype:
        if options.filein.lower().endswith(".lhe") or options.filein.lower().endswith(".lhef") or options.filein.startswith("lhe:"):
            options.filetype="LHE"
        elif options.filein.startswith("mcdb:"):
            print("This is a deprecated way of selecting lhe files from article number. Please use lhe:article argument to --filein")
            options.filein=options.filein.replace('mcdb:','lhe:')
            options.filetype="LHE"
        else:
            options.filetype="EDM"

    filesuffix = {"LHE": "lhe", "EDM": "root", "MCDB": "", "DQM":"root"}[options.filetype]

    if options.filein=="" and not (first_step in ("ALL","GEN","LHE","SIM_CHAIN")):
        options.dirin="file:"+options.dirin.replace('file:','')
        options.filein=trimmedEvtType+"_"+prec_step[first_step]+"."+filesuffix


    # Prepare the canonical file name for output / config file etc
    #   (EventType_STEP1_STEP2_..._PU)
    standardFileName = ""
    standardFileName = trimmedEvtType+"_"+"_".join(options.trimmedStep)
    standardFileName = standardFileName.replace(",","_").replace(".","_")
    if options.pileup != "NoPileUp":
        standardFileName += "_PU"


    # if no output file name given, set it to default
    if options.fileout=="" and not first_step in ("HARVESTING", "ALCAHARVEST"):
        options.fileout = standardFileName+".root"

    # Prepare the name of the config file
    if not options.python_filename:
        options.python_filename = standardFileName+'.py'

    print(options.step)


    # Setting name of process
    # if not set explicitly it needs some thinking
    if not options.name:
        if 'reSIM' in options.trimmedStep:
            options.name = 'RESIM'
        elif 'reDIGI' in options.trimmedStep:
            options.name = 'REDIGI'
        elif 'HLT' in options.trimmedStep:    
            options.name = 'HLT'
        elif 'RECO' in options.trimmedStep:
            options.name = 'RECO'
        elif options.trimmedStep == ['NONE'] and options.filetype in ('LHE', 'MCDB'):
            options.name = 'LHE'
        elif len(options.trimmedStep)==0:
            options.name = 'PROCESS'
        else:
            options.name = options.trimmedStep[-1]

    # check to be sure that people run the harvesting as a separate step
    isHarvesting = False
    isOther = False

    if "HARVESTING" in options.trimmedStep and len(options.trimmedStep) > 1:
        raise Exception("The Harvesting step must be run alone")

    # if not specified by user try to guess whether MC or DATA
    if not options.isData and not options.isMC:
        if 'LHE' in options.trimmedStep or 'LHE' in options.datatier:
            options.isMC=True
        if 'GEN' in options.trimmedStep or 'GEN' in options.datatier:
            options.isMC=True
        if 'SIM' in options.trimmedStep:
            options.isMC=True
        if 'CFWRITER' in options.trimmedStep:
            options.isMC=True
        if 'DIGI' in options.trimmedStep:
            options.isMC=True
        if 'DIGI2RAW' in options.trimmedStep:
            options.isMC=True
        if (not (options.eventcontent == None)) and 'SIM' in options.eventcontent:
            options.isMC=True
        if 'SIM' in options.datatier:
            options.isMC=True
        if options.isMC:
            print('We have determined that this is simulation (if not, rerun cmsDriver.py with --data)')
        else:
            print('We have determined that this is real data (if not, rerun cmsDriver.py with --mc)')

    if options.profile:
        if options.profile and options.prefix:
            raise Exception("--profile and --prefix are incompatible")
        profilerType = 'pp'
        profileOpts = options.profile.split(':')
        if len(profileOpts):
            profilerType = profileOpts[0].replace("=", " ")

        if profilerType == "pp":
            options.profileTypeLabel = "performance"
        elif profilerType == "mp":
            options.profileTypeLabel = "memory"
        elif profilerType.startswith("fp "):
            options.profileTypeLabel = profilerType.replace("fp ", "")
        else:	
            raise Exception("Not a valid profiler type %s. Alternatives are pp, mp, fp=<function>."%(profilerType))

        options.prefix = "igprof -t cmsRun -%s" % profilerType
        
    # If an "era" argument was supplied make sure it is one of the valid possibilities
    if options.era :
        from Configuration.StandardSequences.Eras import eras
        # Split the string by commas to check individual eras
        requestedEras = options.era.split(",")
        # Check that the entry is a valid era
        for eraName in requestedEras :
            if not hasattr( eras, eraName ) or not checkModifier(getattr(eras,eraName)): # Not valid, so print a helpful message
                validOptions="" # Create a stringified list of valid options to print to the user
                for key in eras.__dict__ :
                    if checkModifier(eras.__dict__[key]):
                        if validOptions!="" : validOptions+=", " 
                        validOptions+="'"+key+"'"
                raise Exception( "'%s' is not a valid option for '--era'. Valid options are %s." % (eraName, validOptions) )
    # If the "--fast" option was supplied automatically enable the fastSim era
    if options.fast :
        if options.era:
            options.era+=",fastSim"
        else :
            options.era="fastSim"

    # options incompatible with fastsim
    if options.fast and not options.scenario == "pp":
        raise Exception("ERROR: the --option fast is only compatible with the default scenario (--scenario=pp)")
    if options.fast and 'HLT' in options.trimmedStep:
        raise Exception("ERROR: the --option fast is incompatible with HLT (HLT is no longer available in FastSim)")

    return options

