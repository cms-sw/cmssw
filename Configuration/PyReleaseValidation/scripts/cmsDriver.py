#! /usr/bin/env python

# A Pyrelval Wrapper

import optparse
import sys
import os
import Configuration.PyReleaseValidation
from Configuration.PyReleaseValidation.ConfigBuilder import ConfigBuilder

# Prepare a parser to read the options
usage=\
"""%prog <TYPE> [options].
Examples:
%prog QCD
%prog 10MU+ -e 45 -n 100 --no_output
%prog B_JETS -s DIGI -e 40_130 -n 50 --filein MYSIMJETS --fileout MYDIGIJETS
%prog GAMMA -s DIGI --filein file:myGAMMA.root --dirout rfio:$CASTOR_HOME/test/
%prog GAMMA -s RECO --dirin rfio:$CASTOR_HOME/test/ --fileout file:myGAMMAreco.root
"""
parser = optparse.OptionParser(usage)

parser.add_option("-s", "--step",
                   help="The desired step. The possible values are: "+\
                        "GEN (Generation),"+\
                        "SIM (Simulation), "+\
                        "GENSIM (Generation+Simulation)"+\
                        "DIGI (Digitisation), "+\
                        "RECO (Reconstruction), "+\
                        "ALCA (alignment/calibration), "+\
                        "DIGIRECO (DigitisationReconstruction), "+\
                        "DIGIPURECO (DigitisationReconstruction+ Pileup at low lumi), "+\
                        "ALL (Simulation-Reconstruction-Digitisation).",
                   default="ALL",
                   dest="step")

parser.add_option("-n", "--number",
                   help="The number of evts. The default is 1.",
                   default="1",
                   dest="number")
                   
parser.add_option("--relval",
                   help="Set total number of events and events per job in the ReleaseValidation PSet as <tot_num>,<evts_per_job>.",
                   default="5000,250",
                   dest="relval")                   
                
parser.add_option("--PU",
                  help="Enable the pile up.",
                  action="store_true",
                  default=False,
                  dest="PU_flag")                     
                                     
parser.add_option("--filein",
                   help="The infile name. If absent and necessary a "+\
                        "default value is assigned. "+\
                        "The form is <type>_<energy>_<step>.root.",
                   default="",#to be changed in the default form later
                   dest="filein")

parser.add_option("--secondfilein",
                   help="The secondary infile name."+\
                        "for the two-file solution. Default is no file",
                   default="",#to be changed in the default form later
                   dest="secondfilein")

parser.add_option("--fileout",
                   help="The outfile name. If absent a default value is "+\
                        "assigned. The form is <type>_<energy>_<step>.root.",
                   default="", #to be changed in the default form later
                   dest="fileout")

parser.add_option("--writeraw",
                  help="In addition to the nominal output, write a file with just raw",
                  action="store_true",
                  default=False,
                  dest="writeraw")

parser.add_option("--eventcontent",
                   help="What event content to write out. Default=FEVTDEBUG",
                   default="FEVTDEBUG",
                   dest="eventcontent")

parser.add_option("--datatier",
                   help="What data tier to use. Default from lookup table",
                   default='',
                   dest="datatier")

parser.add_option("--filtername",
                   help="What filter name to specify in output module",
                   default="",
                   dest="filtername")

parser.add_option("--oneoutput",
                   help="use only one output module",
                   action="store_true",
                   default="False",
                   dest="oneoutput")

parser.add_option("--conditions",
                   help="What conditions to use. Default=FrontierConditions_GlobalTag,STARTUP_V4::All",
                   default="FrontierConditions_GlobalTag,STARTUP_V4::All",
                   dest="conditions")

parser.add_option("--beamspot",
                   help="What beam spot to use (from Configuration/StandardSequences). Default=Early10TeVCollision",
                   default="Early10TeVCollision",
                   dest="beamspot")


parser.add_option("--geometry",
                   help="What geometry to use (from Configuration/StandardSequences)",
                   default="",
                   dest="geometry"


                  )

parser.add_option("--magField",
                   help="What magnetic field to use (from Configuration/StandardSequences). Default=3.8T",
                   default="3.8T",
                   dest="magField")

parser.add_option("--altcffs",
                   help="Specify any nondefault cffs to include (replace the default ones) [syntax <step>:cff]",
                   default="",
                   dest="altcffs")

parser.add_option( "--dirin",
                   help="The infile directory.",
                   default="",
                   dest="dirin")                    

parser.add_option( "--dirout",
                   help="The outfile directory.",
                   default="",
                   dest="dirout")                

parser.add_option("-p","--profiler_service",
                  help="Equip the process with the profiler service "+\
                       "by Vincenzo Innocente. First and the last events in "+\
                       " the form <first>_<last>.",
                  default="",
                  dest="profiler_service_cuts")

parser.add_option("--fpe",
                  help="Equip the process with the floating point exception service. "+\
                       "For details see https://twiki.cern.ch/twiki/bin/"+\
                       "view/CMS/SWGuideFloatingPointBehavior",
                  action="store_true",
                  default=False,
                  dest="fpe_service_flag")                        
                  
parser.add_option("--prefix",
                  help="Specify a prefix to the cmsRun command.",
                  default="",
                  dest="prefix")  
                   
parser.add_option("--no_output",
                  help="Do not write anything to disk. This is for "+\
                       "benchmarking purposes.",
                  action="store_true",
                  default=False,
                  dest="no_output_flag")
                                                                                      
parser.add_option("--dump_python",
                  help="Dump the config file in python "+\
                  "language in the file given as argument."+\
                  "If absent and necessary a "+\
                  "default value is assigned. "+\
                  "The form is <type>_<energy>_<step>.py .",
                  action="store_true",
                  default=False,                  
                  dest="dump_python")
                                                    
parser.add_option("--python_filename",
                  help="Change the name of the created config file ",
                  default='',
                  dest="python_filename")

parser.add_option("--dump_DSetName",
                  help="Dump the primary datasetname.",
                  action="store_true",
                  default=False,
                  dest="dump_dsetname_flag")                  
                                    
parser.add_option("--no_exec",
                  help="Do not exec cmsrun. Just prepare the parameters module",
                  action="store_true",
                  default=False,
                  dest="no_exec_flag")   
                  
parser.add_option("--customise",
                  help="Specify the file where the code to modify the process object is stored.",
                  default="",
                  dest="customisation_file")                     

(options,args) = parser.parse_args() # by default the arg is sys.argv[1:]

# A simple check on the consistency of the arguments
if len(sys.argv)==1:
    raise "Event Type: ", "No event type specified!"

options.evt_type=sys.argv[1]

# now adjust the given parameters before passing it to the ConfigBuilder



# Build the IO files if necessary.
# The default form of the files is:
# <type>_<energy>_<step>.root
prec_step = {"ALL":"",
             "GEN":"",
             "SIM":"GEN",
             "DIGI":"SIM",
             "RECO":"DIGI",
             "ALCA":"RECO",
             "ANA":"RECO",
             "DIGI2RAW":"DIGI",
             "RAW2DIGI":"DIGI2RAW"}

trimmedEvtType=options.evt_type.split('/')[-1]

trimmedStep=''
isFirst=0
step_list=options.step.split(',')
for s in step_list:
    stepSP=s.split(':')
    step=stepSP[0]
    if ( isFirst==0 ):
        trimmedStep=step
        isFirst=1
    else:
        trimmedStep=trimmedStep+','+step
        
first_step=trimmedStep.split(',')[0]             
if options.filein=="" and not first_step in ("ALL","GEN","SIM_CHAIN"):
    if options.dirin=="":
        options.dirin="file:"
    options.filein=trimmedEvtType+"_"+prec_step[first_step]+".root"


# Prepare the canonical file name for output / config file etc
#   (EventType_STEP1_STEP2_..._PU)
standardFileName = ""
standardFileName = trimmedEvtType+"_"+trimmedStep
standardFileName = standardFileName.replace(",","_").replace(".","_")
if options.PU_flag:
    standardFileName += "_PU"


# if no output file name given, set it to default
if options.fileout=="":
    options.fileout = standardFileName+".root"


# Prepare the name of the config file
# (in addition list conditions in name)
python_config_filename = standardFileName
conditionsSP = options.conditions.split(',')
if len(conditionsSP) > 1:
    python_config_filename += "_"+str(conditionsSP[1].split("::")[0])
python_config_filename+=".py"


#if desired, just add _rawonly to the end of the output file name
fileraw=''
if options.writeraw:
    fileraw=options.dirout
    wrSP=options.fileout.split('.')
    wrSPLen=len(wrSP)
    counter=0
    for w in wrSP:
        counter=counter+1
        if ( counter < wrSPLen ):
            if ( counter == 1):
                fileraw=fileraw+w
            else:    
                fileraw=fileraw+'.'+w
        else:
            fileraw=fileraw+'_rawonly.'+w

#set process name:
ext_process_name=trimmedEvtType+trimmedStep
options.ext_process_name=trimmedEvtType+trimmedStep


if options.dump_dsetname_flag:
    print ext_process_name
    sys.exit(0) # no need to go further

secondfilestr=''
if options.secondfilein!='':
    secondfilestr=options.dirin+options.secondfilein




# replace step aliases by right list
if options.step=='ALL':
        options.step='GEN,SIM,DIGI,L1,DIGI2RAW,RAW2DIGI,RECO,POSTRECO,DQM'
elif options.step=='DATA_CHAIN':
        options.step='RAW2DIGI,RECO,POSTRECO,DQM'
options.step = options.step.replace("SIM_CHAIN","GEN,SIM,DIGI,L1,DIGI2RAW")



options.name = trimmedStep.replace(',','').replace("_","")
# if we're dealing with HLT, the process name has to be "HLT" only
if 'HLT' in options.name :
    options.name = 'HLT'

options.outfile_name = options.dirout+options.fileout

# after cleanup of all config parameters pass it to the ConfigBuilder
configBuilder = ConfigBuilder(options)
configBuilder.prepare()

# fetch the results and write it to file
if options.python_filename: python_config_filename = options.python_filename
config = file(python_config_filename,"w")
config.write(configBuilder.pythonCfgCode)
config.close()

# handle different dump options
if options.dump_python:
    execfile(python_config_filename, result)
    process = result["process"]
    print "NOT YET IMPLEMENTED"

if options.no_exec_flag:
    print "Config file "+python_config_filename+ " created"
    sys.exit(0)
else:
    commandString = options.prefix+" cmsRun"
    print "Starting "+commandString+' '+python_config_filename
    commands = commandString.lstrip().split()
    os.execvpe(commands[0],commands+[python_config_filename],os.environ)
    sys.exit()
    


    
