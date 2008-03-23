#! /usr/bin/env python

# A Pyrelval Wrapper

import optparse
import sys
import os
import Configuration.PyReleaseValidation

#---------------------------------------------------------

def print_options(options):
    """
    Prints on screen the options specified in the command line.
    """
    opt_dictionary=options.__dict__
    print "\n"
    print "The configuration parameters |-------------"
    opt_dictionary_keys=opt_dictionary.keys()
    opt_dictionary_keys.sort()
    for key in opt_dictionary_keys:
        print key+" "+" "+str(opt_dictionary[key])
    print "-------------------------------------------"

#---------------------------------------------------------

# The supported evt types and default energies:
#energies in GeV!
pgun_ene="10"
jet_en="50_120"
die_en="5_120"
heavy_higgs="190"
light_higgs="120"
ZP_mass="1000"
gravitonmass="1500"
type_energy_dict={"MU+":pgun_ene,
                  "MU-":pgun_ene,
                  "E":pgun_ene,
                  "DIE":die_en,
                  "TAU":pgun_ene,
                  "PI+":pgun_ene,
                  "PI-":pgun_ene,
                  "PI0":pgun_ene,
                  "GAMMA":pgun_ene,
                  #
                  "QCD":"380_470",
                  "TTBAR":"",
                  "MINBIAS":"",           
                  #
                  "B_JETS":jet_en,
                  "C_JETS":jet_en,
                  #
                  "WE":"",
                  "WM":"",
                  "WT":"",
                  #
                  "ZEE":"",
                  "ZMUMU":"",
                  "ZTT":"",
                  #
                  "ZPJJ":"",
                  "ZPEE":ZP_mass,
                  "ZPMUMU":ZP_mass,
                  "ZPTT":ZP_mass,
                  #
                  "HZZEEEE":heavy_higgs,
                  "HZZMUMUMUMU":heavy_higgs,
                  "HZZTTTT":heavy_higgs,
                  "HZZLLLL":heavy_higgs,
                  "HGG":light_higgs,
                  #
                  "RS1GG":gravitonmass,
                  "HpT":""}

# Sorted list of available types for the user help.
types_list=type_energy_dict.keys()
types_list.sort()

# Prepare a parser to read the options
usage=\
"""%prog <TYPE> [options].
The supported event types are: """+str(types_list)+""".
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
                
parser.add_option("-e", "--energy",
                   help="The event energy. If absent, a default value is "+\
                         "assigned according to the event type.",
                   dest="energy") 

parser.add_option("-a","--analysis",
                  help="Enable the analysis.",
                  action="store_true",
                  default=False,
                  dest="analysis_flag")                   

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
                                                                                      
parser.add_option("--dump_cfg",
                  help="Dump the config file in the old config "+\
                       "language. It is printed on stdout.",
                  action="store_true",
                  default=False,
                  dest="dump_cfg")

parser.add_option("--dump_python",
                  help="Dump the config file in python "+\
                  "language in the file given as argument."+\
                  "If absent and necessary a "+\
                  "default value is assigned. "+\
                  "The form is <type>_<energy>_<step>.py .",
                  action="store_true",
                  default=False,                  
                  dest="dump_python")
                                                    
parser.add_option("--dump_pickle",
                  help="Dump a pickle object of the process.",
                  default='',
                  dest="dump_pickle")

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
                  
parser.add_option("--substep3",
                  help="Substitute the \"p3\" sequence with userdefined names.Use ONLY commas to separate values.",
                  default="",
                  dest="newstep3")                                  
                  
parser.add_option("--customise",
                  help="Specify the file where the code to modify the process object is stored.",
                  default="",
                  dest="customisation_file")                     

parser.add_option("--user_schedule",
                  help="User defined schedule instead of the default one.",
                  action="store_true",
                  default=False,
                  dest="user_schedule")

(options,args) = parser.parse_args() # by default the arg is sys.argv[1:]

print options.__dict__

# A simple check on the consistency of the arguments
if len(sys.argv)==1:
    raise "Event Type: ", "No event type specified!"

options.evt_type=sys.argv[1]

#if not options.evt_type in type_energy_dict.keys():
#    raise "Event Type: ","Unrecognised event type."

if options.energy==None:
    if options.evt_type in type_energy_dict.keys():
        options.energy=type_energy_dict[options.evt_type]
    else:
        options.energy=''
        
# Build the IO files if necessary.
# The default form of the files is:
# <type>_<energy>_<step>.root
prec_step = {"ALL":"",
             "GEN":"",
             "SIM":"GEN",
             "DIGI":"SIM",
             "RECO":"DIGI",
             "ANA":"RECO",
             "DIGI2RAW":"DIGI",
             "RAW2DIGI":"DIGI2RAW"}

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
if options.filein=="" and not first_step in ("ALL","GEN"):
    if options.dirin=="":
        options.dirin="file:"
    options.filein=options.evt_type+"_"+options.energy+\
     "_"+prec_step[trimmedStep]+".root"

     
if options.fileout=="":
    options.fileout=options.evt_type+"_"+\
                    options.energy+\
                    "_"+trimmedStep
    if options.PU_flag:
        options.fileout+="_PU"
    if options.analysis_flag:
        options.fileout+="_ana"    
    options.fileout+=".root"

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

# File where to dump the python cfg file
python_config_filename=''
if options.dump_python:
    python_config_filename=options.evt_type+"_"+\
                              options.energy+\
                              "_"+trimmedStep
    if options.PU_flag:
        python_config_filename+="_PU"
    if options.analysis_flag:
        python_config_filename+="_ana"
    python_config_filename+=".py"

cfg_config_filename=''
if options.dump_cfg:
    cfg_config_filename=options.evt_type+"_"+\
                              options.energy+\
                              "_"+trimmedStep
    if options.PU_flag:
        cfg_config_filename+="_PU"
    if options.analysis_flag:
        cfg_config_filename+="_ana"
    cfg_config_filename+=".cfg"
    
#prepare new step3 list:
newstep3list=[]
if options.newstep3!="":
    newstep3list=options.newstep3.split(",")    

# Print the options to screen
if not options.dump_dsetname_flag:
    print_options(options)  

#set process name:
ext_process_name=options.evt_type+options.energy+trimmedStep


if options.dump_dsetname_flag:
    print ext_process_name
    sys.exit(0) # no need to go further
    
cfgfile="""
#############################################################
#                                                           #
#             + relval_parameters_module +                  #
#                                                           #
#  The supported types are:                                 #
#                                                           #
#   - QCD (energy in the form min_max)                      #
#   - B_JETS, C_JETS, UDS_JETS (energy in the form min_max) #
#   - TTBAR                                                 #
#   - BSJPSIPHI                                             #
#   - MU+,MU-,E+,E-,GAMMA,10MU+,10E-...                     #
#   - TAU (energy in the form min_max for cuts)             #
#   - HZZEEEE, HZZMUMUMUMU                                  #
#   - ZEE (no energy is required)                           #
#   - ZPJJ: zee prime in 2 jets                             #
#                                                           #
#############################################################

# Process Parameters

# The name of the process
process_name='""" +trimmedStep.replace(',','')+ """'
ext_process_name='""" +ext_process_name+ """'
# The type of the process. Please see the complete list of 
# available processes.
evt_type='"""+options.evt_type+"""'
# The energy in GeV. Some of the tipes require an
# energy in the form "Emin_Emax"
energy='"""+options.energy+"""'
# The PU
PU_flag="""+str(options.PU_flag)+"""
# Number of evts to generate
evtnumber="""+options.number+"""
# The ReleaseValidation PSet
releasevalidation=("""+options.relval+""")
# Input and output file names
infile_name='"""+options.dirin+options.filein+"""'
outfile_name='"""+options.dirout+options.fileout+"""'
rawfile_name='"""+fileraw+"""'
# The step
step='"""+str(options.step)+"""'
# Omit the output in a root file
output_flag="""+str(not options.no_output_flag)+"""
# Use the profiler service
profiler_service_cuts='"""+options.profiler_service_cuts+"""'
# Use the floating point exception module:
fpe_service_flag="""+str(options.fpe_service_flag)+"""
# Substitute Step 3 sequence
newstep3list="""+str(newstep3list)+"""
# The anlaysis
analysis_flag="""+str(options.analysis_flag)+"""
# Customisation_file
customisation_file='"""+str(options.customisation_file)+"""'
# User defined schedule
user_schedule="""+str(options.user_schedule)+"""

# Pyrelval parameters
# Enable verbosity
dbg_flag=True
# Dump the oldstyle cfg file.
dump_cfg='"""+cfg_config_filename+"""'
# Dump the python cfg file.
dump_python='"""+python_config_filename+"""'
# Dump a pickle object of the process on disk.
dump_pickle='"""+str(options.dump_pickle)+"""'
#Dump the dataset Name
dump_dsetname_flag="""+str(options.dump_dsetname_flag)+"""

"""

# Write down the configuration in a Python module
config_module_name="./relval_parameters_module.py" 
config_module=file(config_module_name,"w")
config_module.write(cfgfile)
config_module.close()

# Prepare command execution
cmssw_base=os.environ["CMSSW_BASE"]
cmssw_release_base=os.environ["CMSSW_RELEASE_BASE"]
pyrelvallocal=cmssw_base+"/src/Configuration/PyReleaseValidation"
#Set the path depending on the presence of a locally checked out version of PyReleaseValidation
if os.path.exists(pyrelvallocal):
    # set the PYTHONPATH environmental variable
    pyrelvalcodedir=cmssw_base+"/src/Configuration/PyReleaseValidation/data/"
    print "Using LOCAL version of Configuration/PyReleaseValidation instead of the RELEASE version"
elif not os.path.exists(pyrelvallocal):
    pyrelvalcodedir=cmssw_release_base+"/src/Configuration/PyReleaseValidation/data/"
os.environ["PYTHONPATH"]+=":"+pyrelvalcodedir

executable='cmsRun'
if options.dump_pickle!='':
    executable='python'

command=['/bin/sh', '-c', 'exec ']
pyrelvalmain="`which relval_main.py`"
if options.prefix!="": 
    command[2] += options.prefix + ' '
command[2] += executable + ' ' + pyrelvalmain
sys.stdout.flush() 

# And Launch the Framework or just dump the parameters module
if options.no_exec_flag:
    config_module=file(config_module_name,"r")
    print config_module.read()
    config_module.close()
    print "Parameters module created."
    sys.exit(0) # Exits without launching cmsRun


# Remove existing pyc files:
os.system("rm -f "+pyrelvalcodedir+"*.pyc")    
# A temporary ugly fix for a problem to investigate further.

print "Launching "+' '.join(command)+"..."
os.execvpe(command[0], command, os.environ) # Launch
    
