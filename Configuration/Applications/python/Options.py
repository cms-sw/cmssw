#! /usr/bin/env python3

# A Pyrelval Wrapper

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import sys
import os
import re
import Configuration.Applications
from Configuration.Applications.ConfigBuilder import ConfigBuilder, defaultOptions
import traceback
# Prepare a parser to read the options
usage=\
"""%(prog)s [options].
Example:

%(prog)s -s RAW2DIGI,RECO --conditions STARTUP_V4::All --eventcontent RECOSIM
"""
parser = ArgumentParser(usage=usage, formatter_class=ArgumentDefaultsHelpFormatter)

expertSettings = parser.add_argument_group('===============\n  Expert Options', 'Caution: please use only if you know what you are doing.')

parser.add_argument("-s", "--step",
                    help="The desired step. The possible values are: "+\
                         "GEN,SIM,DIGI,L1,DIGI2RAW,HLT,RAW2DIGI,RECO,POSTRECO,DQM,ALCA,VALIDATION,HARVESTING, NONE or ALL.",
                    default="ALL",
                    type=str,
                    dest="step")

parser.add_argument("--conditions",
                    help="What conditions to use (required; provide value 'help' to get list of options)",
                    required=True,
                    type=str,
                    dest="conditions")

parser.add_argument("--eventcontent",
                    help="What event content to write out",
                    default='RECOSIM',
                    type=str,
                    dest="eventcontent")

parser.add_argument("--filein",
                    help="The infile name.",
                    default="", #to be changed in the default form later
                    type=str,
                    dest="filein")

parser.add_argument("--fileout",
                    help="The outfile name. If absent a default value is assigned",
                    default="", #to be changed in the default form later
                    type=str,
                    dest="fileout")

parser.add_argument("--filetype",
                    help="The type of the infile",
                    default=defaultOptions.filetype,
                    type=str,
                    dest="filetype",
                    choices=['EDM','DAT','LHE','MDCB','DQM','DQMDAQ']
                  )

parser.add_argument("-n", "--number",
                    help="The number of events.",
                    default=1,
                    type=int,
                    dest="number")

parser.add_argument("-o", "--number_out",
                    help="The number of events in output.",
                    default=None,
                    type=int,
                    dest="number_out")

parser.add_argument("--mc",
                    help="Specify that simulation is to be processed (default = guess based on options)",
                    action="store_true",
                    default=False,
                    dest="isMC")

parser.add_argument("--data",
                    help="Specify that data is to be processed (default = guess based on options)",
                    action="store_true",
                    default=False,
                    dest="isData")

parser.add_argument("--no_exec",
                    help="Do not exec cmsRun. Just prepare the python config file.",
                    action="store_true",
                    default=False,
                    dest="no_exec_flag")

parser.add_argument("--fast",
                    help="Specify that the configuration is for FASTSIM",
                    action="store_true",
                    default=False)

parser.add_argument("--runsAndWeightsForMC",
                    help="Assign run numbers to MC source according to relative weights. [(run1,weight1),...,(runN,weightN)])",
                    default=None,
                    dest="runsAndWeightsForMC")

parser.add_argument("--runsScenarioForMC",
                    help="Load a scenario to set run numbers in MC.)",
                    default=None,
                    dest="runsScenarioForMC")

parser.add_argument("--runsAndWeightsForMCIntegerWeights",
                    help="Assign run numbers to MC source according to relative weights where weighting is determined by the number of times the run number appears. [(run1,run2,...,runN)])",
                    default=None,
                    dest="runsAndWeightsForMCIntegerWeights")

parser.add_argument("--runsScenarioForMCIntegerWeights",
                    help="Load a scenario to set run numbers in MC with integer IOV weights.",
                    default=None,
                    dest="runsScenarioForMCIntegerWeights")

parser.add_argument("--runUnscheduled",
                    help="Automatically convert configuration to run unscheduled the EDProducers/EDFilters that were scheduled",
                    action="store_true",
                    default=False,
                    dest="runUnscheduled")

# expert settings
expertSettings.add_argument("--beamspot",
                            help="What beam spot to use (from Configuration/StandardSequences). Default depends on scenario",
                            default=None,
                            type=str,
                            dest="beamspot")

expertSettings.add_argument("--customise",
                            help="Specify the file where the code to modify the process object is stored.",
                            default=[],
                            action="append",
                            type=str,
                            dest="customisation_file")

expertSettings.add_argument("--customise_unsch",
                            help="Specify the file where the code to modify the process object is stored.",
                            default=[],
                            action="append",
                            type=str,
                            dest="customisation_file_unsch")

expertSettings.add_argument("--customise_commands",
                            help="Specify a string of commands",
                            default="",
                            type=str,
                            dest="customise_commands")

expertSettings.add_argument("--inline_custom",
                            help="inline the customisation file",
                            default=False,
                            action="store_true",
                            dest="inline_custom")

expertSettings.add_argument("--datatier",
                            help="What data tier to use.",
                            default='',
                            type=str,
                            dest="datatier")

expertSettings.add_argument( "--dirin",
                            help="The infile directory.",
                            default="",
                            type=str,
                            dest="dirin")

expertSettings.add_argument( "--dirout",
                            help="The outfile directory.",
                            default="",
                            type=str,
                            dest="dirout")

expertSettings.add_argument("--filtername",
                            help="What filter name to specify in output module",
                            default="",
                            type=str,
                            dest="filtername")

expertSettings.add_argument("--geometry",
                            help="What simulation geometry to use. Comma-separated SimGeometry,RecoGeometry is supported.",
                            default=defaultOptions.geometry,
                            type=str,
                            dest="geometry")

expertSettings.add_argument("--magField",
                            help="What magnetic field to use (from Configuration/StandardSequences).",
                            default=defaultOptions.magField,
                            type=str,
                            dest="magField")

expertSettings.add_argument("--no_output",
                            help="Do not write anything to disk. This is for "+\
                            "benchmarking purposes.",
                            action="store_true",
                            default=False,
                            dest="no_output_flag")

expertSettings.add_argument("--prefix",
                            help="Specify a prefix to the cmsRun command.",
                            default="",
                            type=str,
                            dest="prefix")

expertSettings.add_argument("--suffix",
                            help="Specify a suffix to the cmsRun command.",
                            default="",
                            type=str,
                            dest="suffix")

expertSettings.add_argument("--relval",
                            help="Set total number of events and events per job.", #this does not get used but get parsed in the command by DataOps
                            default="",
                            dest="relval")

expertSettings.add_argument("--dump_python",
                            help="Dump the config file in python "+\
                            "and do a full expansion of imports.",
                            action="store_true",
                            default=False,
                            dest="dump_python")

expertSettings.add_argument("--pileup",
                            help="What pileup config to use",
                            default=defaultOptions.pileup,
                            type=str,
                            dest="pileup")
    
expertSettings.add_argument("--pileup_input",
                            help="define the pile up files to mix with",
                            default=None,
                            type=str,
                            dest="pileup_input")

expertSettings.add_argument("--pileup_dasoption",
                            help="Additional option for DAS query of pile up",
                            default="",
                            type=str,
                            dest="pileup_dasoption")

expertSettings.add_argument("--datamix",
                            help="What datamix config to use",
                            default=defaultOptions.datamix,
                            type=str,
                            dest="datamix")

expertSettings.add_argument("--gflash",
                            help="Run the FULL SIM using the GFlash parameterization.",
                            action="store_true",
                            default=defaultOptions.gflash,
                            dest="gflash")

expertSettings.add_argument("--python_filename",
                            help="Change the name of the created config file",
                            default='',
                            type=str,
                            dest="python_filename")

expertSettings.add_argument("--secondfilein",
                            help="The secondary infile name."+\
                                "for the two-file solution. Default is no file",
                            default="", #to be changed in the default form later
                            type=str,
                            dest="secondfilein")

expertSettings.add_argument("--processName",
                            help="set process name explicitly",
                            default = None,
                            type=str,
                            dest="name")

expertSettings.add_argument("--triggerResultsProcess",
                            help="for splitting jobs specify from which process to take edm::TriggerResults",
                            default = None,
                            type=str,
                            dest="triggerResultsProcess")

expertSettings.add_argument("--hltProcess",
                            help="modify the DQM sequence to look for HLT trigger results with the specified process name",
                            default = None,
                            type=str,
                            dest="hltProcess")

expertSettings.add_argument("--scenario",
                            help="Select scenario overriding standard settings",
                            default='pp',
                            type=str,
                            dest="scenario",
                            choices=defaultOptions.scenarioOptions)

expertSettings.add_argument("--harvesting",
                            help="What harvesting to use (from Configuration/StandardSequences)",
                            default=defaultOptions.harvesting,
                            type=str,
                            dest="harvesting")

expertSettings.add_argument("--particle_table",
                            help="Which particle properties table is loaded",
                            default=defaultOptions.particleTable,
                            type=str,
                            dest="particleTable")

expertSettings.add_argument("--dasquery",
                            help="Allow to define the source.fileNames from the das search command",
                            default='',
                            type=str,
                            dest="dasquery")

expertSettings.add_argument("--dasoption",
                            help="Additional option for DAS query",
                            default='',
                            type=str,
                            dest="dasoption")

expertSettings.add_argument("--dbsquery",
                            help="Deprecated. Please use dasquery option. Functions for backward compatibility",
                            default='',
                            type=str,
                            dest="dasquery")

expertSettings.add_argument("--lazy_download",
                            help="Enable lazy downloading of input files",
                            action="store_true",
                            default=False,
                            dest="lazy_download")

expertSettings.add_argument("--repacked",
                            help="When the input file is a file with repacked raw data with label rawDataRepacker",
                            action="store_true",
                            default=False,
                            dest="isRepacked")

expertSettings.add_argument("--custom_conditions",
                            help="Allow to give a few overriding tags for the GT",
                            default='',
                            type=str,
                            dest='custom_conditions')

expertSettings.add_argument("--inline_eventcontent",
                            help="expand event content definitions",
                            action="store_true",
                            default=False,
                            dest="inlineEventContent")

expertSettings.add_argument("--inline_object",
                            help="expand explicitly the definition of a list of objects",
                            default='',
                            type=str,
                            dest="inlineObjects")

expertSettings.add_argument("--hideGen",
                            help="do not inline the generator information, just load it",
                            default=False,
                            action="store_true")

expertSettings.add_argument("--output",
                            help="specify the list of output modules using dict",
                            default='',
                            type=str,
                            dest="outputDefinition")

expertSettings.add_argument("--inputCommands",
                            help="specify the input commands; i.e dropping products",
                            default=None,
                            type=str,
                            dest="inputCommands")

expertSettings.add_argument("--outputCommands",
                            help="specify the extra output commands;",
                            default=None,
                            type=str,
                            dest="outputCommands")

expertSettings.add_argument("--inputEventContent",
                            help="specify the input event content",
                            default=defaultOptions.inputEventContent,
                            type=str,
                            dest="inputEventContent")

expertSettings.add_argument("--dropDescendant",
                            help="allow to drop descendant on input",
                            default=defaultOptions.dropDescendant,
                            action="store_true")

expertSettings.add_argument("--donotDropOnInput",
                            help="when using reSTEP, prevent the automatic product dropping on input",
                            default=defaultOptions.donotDropOnInput,
                            type=str)

# specifying '--restoreRNDSeeds' results in 'options.restoreRNDSeeds = True'
# specifying '--restoreRNDSeeds arg' results in 'options.restoreRNDSeeds = arg'
expertSettings.add_argument("--restoreRNDSeeds",
                            help="restore the random number engine state",
                            default=False,
                            const=True,
                            type=str,
                            nargs='?')

expertSettings.add_argument("--era",
                            help="Specify which era to use (e.g. \"run2\")",
                            default=None,
                            type=str,
                            dest="era")

expertSettings.add_argument("--procModifiers",
                            help="Specify any process Modifiers to include (in Configuration/ProcessModiers) - comma separated list",
                            default=[],
                            action="append",
                            type=str,
                            dest="procModifiers")

expertSettings.add_argument("--evt_type",
                            help="specify the gen fragment",
                            default=None,
                            type=str,
                            dest="evt_type")

expertSettings.add_argument("--profile",
                            help="add the IgprofService with the parameter provided PROFILER:START:STEP:PEREVENOUTPUTFORMAT:ENDOFJOBOUTPUTFORMAT",
                            default=None,
                            type=str,
                            dest="profile")

expertSettings.add_argument("--heap_profile",
                            help="add the JeProfService with the parameter provided PROFILER:START:STEP:PEREVENOUTPUTFORMAT:ENDOFJOBOUTPUTFORMAT",
                            default=None,
                            type=str,
                            dest="heap_profile")

expertSettings.add_argument("--io",
                            help="Create a json file with io informations",
                            default=None,
                            type=str,
                            dest="io")

expertSettings.add_argument("--lumiToProcess",
                            help="specify a certification json file in input to run on certified data",
                            default=None,
                            type=str,
                            dest='lumiToProcess')

expertSettings.add_argument("--timeoutOutput",
                            help="use a TimeoutPoolOutputModule instead of a PoolOutputModule (needed for evt. display)",
                            default=False,
                            action="store_true",
                            dest='timeoutOutput')

expertSettings.add_argument("--nThreads",
                            help="How many threads should CMSSW use",
                            default=defaultOptions.nThreads,
                            type=int,
                            dest='nThreads')

expertSettings.add_argument("--nStreams",
                            help="How many streams should CMSSW use if nThreads > 1 (default is 0 which makes it same as nThreads)",
                            default=defaultOptions.nStreams,
                            type=int,
                            dest='nStreams')

expertSettings.add_argument("--nConcurrentLumis",
                            help="How many concurrent LuminosityBlocks should CMSSW use if nThreads > 1 (default is 0 which means 1 for 1 stream and 2 for >= 2 streams)",
                            default=defaultOptions.nConcurrentLumis,
                            type=int,
                            dest='nConcurrentLumis')

expertSettings.add_argument("--nConcurrentIOVs",
                            help="How many concurrent IOVs should CMSSW use if nThreads > 1",
                            default=defaultOptions.nConcurrentIOVs,
                            type=int,
                            dest='nConcurrentIOVs')

expertSettings.add_argument("--accelerators",
                            help="Comma-separated list of accelerators to enable; if 'cpu' is not included, the job will fail if none of the accelerators is available (default is not set, enabling all available accelerators, including the cpu)",
                            default=None,
                            type=str,
                            dest='accelerators')
