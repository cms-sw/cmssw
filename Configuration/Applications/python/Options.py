#! /usr/bin/env python3

# A Pyrelval Wrapper

import optparse
import sys
import os
import re
import Configuration.Applications
from Configuration.Applications.ConfigBuilder import ConfigBuilder, defaultOptions
import traceback
# Prepare a parser to read the options
usage=\
"""%prog <TYPE> [options].
Example:

%prog reco -s RAW2DIGI,RECO --conditions STARTUP_V4::All --eventcontent RECOSIM
"""
parser = optparse.OptionParser(usage)

expertSettings = optparse.OptionGroup(parser, '===============\n  Expert Options', 'Caution: please use only if you know what you are doing.')
famosSettings = optparse.OptionGroup(parser, '===============\n  FastSimulation options', '')
parser.add_option_group(expertSettings)

threeValued=[]
parser.add_option("-s", "--step",
                   help="The desired step. The possible values are: "+\
                        "GEN,SIM,DIGI,L1,DIGI2RAW,HLT,RAW2DIGI,RECO,POSTRECO,DQM,ALCA,VALIDATION,HARVESTING, NONE or ALL.",
                   default="ALL",
                   dest="step")

parser.add_option("--conditions",
                  help="What conditions to use. This has to be specified",
                  default=None,
                  dest="conditions")

parser.add_option("--eventcontent",
                  help="What event content to write out. Default=FEVTDEBUG, or FEVT (for cosmics)",
                  default='RECOSIM',
                  dest="eventcontent")

parser.add_option("--filein",
                   help="The infile name.",
                   default="",#to be changed in the default form later
                   dest="filein")

parser.add_option("--fileout",
                   help="The outfile name. If absent a default value is assigned",
                   default="", #to be changed in the default form later
                   dest="fileout")

parser.add_option("--filetype",
                  help="The type of the infile (EDM, LHE or MCDB).",
                  default=defaultOptions.filetype,
                  dest="filetype",
                  choices=['EDM','DAT','LHE','MDCB','DQM','DQMDAQ']
                  )

parser.add_option("-n", "--number",
                  help="The number of events. The default is 1.",
                  default="1",
                  dest="number")
parser.add_option("-o", "--number_out",
                  help="The number of events in output. The default is not set",
                  default=None,
                  dest="number_out")

parser.add_option("--mc",
                  help="Specify that simulation is to be processed (default = guess based on options",
                  action="store_true",
                  default=False,
                  dest="isMC")

parser.add_option("--data",
                  help="Specify that data is to be processed (default = guess based on options",
                  action="store_true",
                  default=False,
                  dest="isData")


parser.add_option("--no_exec",
                  help="Do not exec cmsRun. Just prepare the python config file.",
                  action="store_true",
                  default=False,
                  dest="no_exec_flag")   
parser.add_option("--fast",
                  help="Specify that the configuration is for FASTSIM",
                  action="store_true",
                  default=False)

parser.add_option("--runsAndWeightsForMC",
                  help="Assign run numbers to MC source according to relative weights. [(run1,weight1),...,(runN,weightN)])",
                  default=None,
                  dest="runsAndWeightsForMC")

parser.add_option("--runsScenarioForMC",
                  help="Load a scenario to set run numbers in MC.)",
                  default=None,
                  dest="runsScenarioForMC")

parser.add_option("--runsAndWeightsForMCIntegerWeights",
                  help="Assign run numbers to MC source according to relative weights where weighting is determined by the number of times the run number appears. [(run1,run2,...,runN)])",
                  default=None,
                  dest="runsAndWeightsForMCIntegerWeights")

parser.add_option("--runsScenarioForMCIntegerWeights",
                  help="Load a scenario to set run numbers in MC with integer IOV weights.)",
                  default=None,
                  dest="runsScenarioForMCIntegerWeights")

parser.add_option("--runUnscheduled",
                  help="Automatically convert configuration to run unscheduled the EDProducers/EDFilters that were scheduled",
                  action="store_true",
                  default=False,
                  dest="runUnscheduled")

# expert settings
expertSettings.add_option("--beamspot",
                          help="What beam spot to use (from Configuration/StandardSequences). Default depends on scenario",
                          default=None,
                          dest="beamspot")

expertSettings.add_option("--customise",
                          help="Specify the file where the code to modify the process object is stored.",
                          default=[],
                          action="append",
                          dest="customisation_file")
expertSettings.add_option("--customise_unsch",
                          help="Specify the file where the code to modify the process object is stored.",
                          default=[],
                          action="append",
                          dest="customisation_file_unsch")
expertSettings.add_option("--customise_commands",
                          help="Specify a string of commands",
                          default="",
                          dest="customise_commands")

expertSettings.add_option("--inline_custom",
                          help="inline the customisation file",
                          default=False,
                          action="store_true",
                          dest="inline_custom")

expertSettings.add_option("--datatier",
                          help="What data tier to use.",
                          default='',
                          dest="datatier")

expertSettings.add_option( "--dirin",
                          help="The infile directory.",
                          default="",
                          dest="dirin")                    

expertSettings.add_option( "--dirout",
                          help="The outfile directory.",
                          default="",
                          dest="dirout")                

expertSettings.add_option("--filtername",
                          help="What filter name to specify in output module",
                          default="",
                          dest="filtername")

expertSettings.add_option("--geometry",
                          help="What simulation geometry to use. Default="+defaultOptions.geometry+". Coma separated SimGeometry,RecoGeometry is supported.",
                          default=defaultOptions.geometry,
                          dest="geometry")

expertSettings.add_option("--magField",
                          help="What magnetic field to use (from Configuration/StandardSequences).",
                          default=defaultOptions.magField,
                          dest="magField")

expertSettings.add_option("--no_output",
                          help="Do not write anything to disk. This is for "+\
                          "benchmarking purposes.",
                          action="store_true",
                          default=False,
                          dest="no_output_flag")

expertSettings.add_option("--prefix",
                          help="Specify a prefix to the cmsRun command.",
                          default="",
                          dest="prefix")

expertSettings.add_option("--suffix",
                          help="Specify a suffix to the cmsRun command.",
                          default="",
                          dest="suffix")  

expertSettings.add_option("--relval",
                          help="Set total number of events and events per job.", #this does not get used but get parsed in the command by DatOps
                          default="",
                          dest="relval")

expertSettings.add_option("--dump_python",
                  help="Dump the config file in python "+\
                  "and do a full expansion of imports.",
                  action="store_true",
                  default=False,                  
                  dest="dump_python")

expertSettings.add_option("--pileup",
                  help="What pileup config to use. Default="+defaultOptions.pileup,
                  default=defaultOptions.pileup,
                  dest="pileup")
    
expertSettings.add_option("--pileup_input",
                          help="define the pile up files to mix with",
                          default=None,
                          dest="pileup_input")

expertSettings.add_option("--pileup_dasoption",
                          help="Additional option for DAS query of pile up",
                          default="",
                          dest="pileup_dasoption")

expertSettings.add_option("--datamix",
                  help="What datamix config to use. Default=DataOnSim.",
                  default=defaultOptions.datamix,
                  dest="datamix")

expertSettings.add_option("--gflash",
                  help="Run the FULL SIM using the GFlash parameterization.",
                  action="store_true",
                  default=defaultOptions.gflash,
                  dest="gflash")

expertSettings.add_option("--python_filename",
                          help="Change the name of the created config file ",
                          default='',
                          dest="python_filename")

expertSettings.add_option("--secondfilein",
                          help="The secondary infile name."+\
                                "for the two-file solution. Default is no file",
                          default="",#to be changed in the default form later
                          dest="secondfilein")

expertSettings.add_option("--processName",
                          help="set process name explicitly",
                          default = None,
                          dest="name" 
                          )

expertSettings.add_option("--triggerResultsProcess",
                          help="for splitting jobs specify from which process to take edm::TriggerResults",
                          default = None,
                          dest="triggerResultsProcess"
                          )

expertSettings.add_option("--hltProcess",
                          help="modify the DQM sequence to look for HLT trigger results with the specified process name", 
                          default = None,
                          dest="hltProcess"
                          )

expertSettings.add_option("--scenario",
                          help="Select scenario overriding standard settings (available:"+str(defaultOptions.scenarioOptions)+")",
                          default='pp',
                          dest="scenario",
                          choices=defaultOptions.scenarioOptions)

expertSettings.add_option("--harvesting",
                          help="What harvesting to use (from Configuration/StandardSequences). Default=AtRunEnd",
                          default=defaultOptions.harvesting,
                          dest="harvesting")

expertSettings.add_option("--particle_table",
                          help="Which particle properties table is loaded. Default=pythia",
                          default=defaultOptions.particleTable,
                          dest="particleTable")

expertSettings.add_option("--dasquery",
                          help="Allow to define the source.fileNames from the das search command",
                          default='',
                          dest="dasquery")

expertSettings.add_option("--dasoption",
                          help="Additional option for DAS query",
                          default='',
                          dest="dasoption")

expertSettings.add_option("--dbsquery",
                          help="Deprecated. Please use dasquery option. Functions for backward compatibility",
                          default='',
                          dest="dasquery")

expertSettings.add_option("--lazy_download",
                  help="Enable lazy downloading of input files",
                  action="store_true",
                  default=False,
                  dest="lazy_download")   

expertSettings.add_option("--repacked",
                          help="When the input file is a file with repacked raw data with label rawDataRepacker",
                          action="store_true",
                          default=False,
                          dest="isRepacked"
                          )

expertSettings.add_option("--custom_conditions",
                          help="Allow to give a few overriding tags for the GT",
                          default='',
                          dest='custom_conditions')

expertSettings.add_option("--inline_eventcontent",
                          help="expand event content definitions",
                          action="store_true",
                          default=False,
                          dest="inlineEventContent")


expertSettings.add_option("--inline_object",
                          help="expand explicitely the definition of a list of objects",
                          default='',
                          dest="inlineObjets")

expertSettings.add_option("--hideGen",
                          help="do not inline the generator information, just load it",
                          default=False,
                          action="store_true")
expertSettings.add_option("--output",
                          help="specify the list of output modules using dict",
                          default='',
                          dest="outputDefinition")

expertSettings.add_option("--inputCommands",
                          help="specify the input commands; i.e dropping products",
                          default=None,
                          dest="inputCommands")
expertSettings.add_option("--outputCommands",
                          help="specify the extra output commands;",
                          default=None,
                          dest="outputCommands")

expertSettings.add_option("--inputEventContent",
                          help="specify the input event content",
                          default=defaultOptions.inputEventContent,
                          dest="inputEventContent")

expertSettings.add_option("--dropDescendant",
                          help="allow to drop descendant on input",
                          default=defaultOptions.dropDescendant,
                          action="store_true")

expertSettings.add_option("--donotDropOnInput",
                          help="when using reSTEP, prevent the automatic product dropping on input",
                          default=defaultOptions.donotDropOnInput
                          )

expertSettings.add_option("--restoreRNDSeeds",
                          help="restore the random number engine state",
                          default=False,
                          )
threeValued.append( ('--restoreRNDSeeds',True) )


expertSettings.add_option("--era",
                          help="Specify which era to use (e.g. \"run2\")",
                          default=None,
                          dest="era")

expertSettings.add_option("--procModifiers",
                          help="Specify any process Modifiers to include (in Configuration/ProcessModiers) - comma separated list",
                          default=None,
                          dest="procModifiers")

expertSettings.add_option("--evt_type",
                          help="specify the gen fragment",
                          default=None,
                          dest="evt_type")

expertSettings.add_option("--profile",
                          help="add the IgprofService with the parameter provided PROFILER:START:STEP:PEREVENOUTPUTFORMAT:ENDOFJOBOUTPUTFORMAT",
                          default=None,
                          dest="profile")

expertSettings.add_option("--io",
                          help="Create a json file with io informations",
                          default=None,
                          dest="io")

expertSettings.add_option("--lumiToProcess",
                          help="specify a certification json file in input to run on certified data",
                          default=None,
                          dest='lumiToProcess'
                          )

expertSettings.add_option("--timeoutOutput",
                          help="use a TimeoutPoolOutputModule instead of a PoolOutputModule (needed for evt. display)",
                          default=False,
                          dest='timeoutOutput'
                          )

expertSettings.add_option("--nThreads",
                          help="How many threads should CMSSW use (default is 1)",
                          default=defaultOptions.nThreads,
                          dest='nThreads'
                          )
expertSettings.add_option("--nStreams",
                          help="How many streams should CMSSW use if nThreads > 1 (default is 0 which makes it same as nThreads)",
                          default=defaultOptions.nStreams,
                          dest='nStreams'
                          )
expertSettings.add_option("--nConcurrentLumis",
                          help="How many concurrent LuminosityBlocks should CMSSW use if nThreads > 1 (default is 0 which means 1 for 1 stream and 2 for >= 2 streams)",
                          default=defaultOptions.nConcurrentLumis,
                          dest='nConcurrentLumis'
                          )
expertSettings.add_option("--nConcurrentIOVs",
                          help="How many concurrent IOVs should CMSSW use if nThreads > 1 (default is 1)",
                          default=defaultOptions.nConcurrentIOVs,
                          dest='nConcurrentIOVs'
                          )
