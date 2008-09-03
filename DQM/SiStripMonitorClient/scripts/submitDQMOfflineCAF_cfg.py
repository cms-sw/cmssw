#!/usr/bin/env python

#
# $$
#

## CMSSW/DQM/SiStripMonitorClient/scripts/submitDQMOfflineCAF_cfg.py
#
#  This script submits batch jobs to the CAF in order to process the full
#  granularity SiStrip offline DQM.
#  Questions and comments to: volker.adler@cern.ch


import sys
import os
import os.path
import commands
import shutil
import string
import math
import urllib
import time

# Constants

# numbers
OCT_rwx_r_r = 0744
# strings
STR_default              = 'DEFAULT'
STR_nameInputFilesJobCff = 'inputFiles.cff'
STR_nameCmsswPackage     = 'DQM/SiStripMonitorClient'
STR_magField0            = '0.0'
STR_magField20           = '2.0'
STR_magField30           = '3.0'
STR_magField35           = '3.5'
STR_magField38           = '3.8'
STR_magField40           = '4.0'
STR_textUsage            = """ CMSSW/DQM/SiStripMonitorClient/scripts/submitDQMOfflineCAF_cfg.py
 
 This script submits batch jobs to the CAF in order to process the full
 granularity SiStrip offline DQM.
 Questions and comments to: volker.adler@cern.ch
 
 Usage(): submitDQMOfflineCAF_cfg.py (-s, --submit | -c, --create |
                                      -h, --help)
                                     [-r, --run]
                                     [-C, --CRAB]
                                     [-S, --server]
                                     [-e, --email]
                                     [-j, --jobs]
                                     [-f, --filter]
                                     [-d, --dataset]
                                     [-o, --outpath]
                                     [-m, --mergepath]                               
                               
   Function letters: One of the following options  m u s t  be used.
   
     -s, --submit
         create jobs and submit them to CAF;
         requires option '-r'
         
     -c, --create
         create jobs, but do not submit them;
         requires option '-r'
         
     -h, --help
         print this message
         
   Other options:
   
     -r, --run RUNNUMBER
         number of run to process;
         required by funtion letters '-s' and '-c'
   
     -C, --CRAB TRUE/FALSE
         submit or submit not using CRAB;
         default: TRUE
   
     -S, --server CRAB_SERVER
         CRAB server to use;
         available: None (default)
                    caf
                    bari (currently not usable for submission due to drain mode,
                          s. https://twiki.cern.ch/twiki/bin/view/CMS/CrabServer#Server_available_for_users)
                    lnl2
                    
         NOTE: CRAB server submission is disabled at the moment.
         
     -e, --email EMAIL_ADDRESS
         where the CRAB server should send its messages;
         default: volker.adler@cern.ch
         
     -j, --jobs NUMBER
         number of jobs to create;
         default: 10
         
     -f, --filter TRUE/FALSE
         use or use not HLT filters to select events to process;
         default: FALSE
      
     -d, --dataset PRIMARY_DATASET
         specify dataset for DBS query;
         available: /Cosmics/EW35_3T_v1_CRUZET4_V3P_SuperPointing_v1/RECO
                    /Cosmics/EW35_3T_v1_CRUZET4_V3P_TrackerPointing_v1/RECO
                    /Cosmics/Commissioning08-EW35_3T_v1/RECO                         (default)
                    /Cosmics/Commissioning08-EW35_3T_v1/RAW
                    /Cosmics/CRUZET4_v1_CRZT210_V1_SuperPointing_v1/RECO
                    /Cosmics/CRUZET4_v1_CRZT210_V1_TrackerPointing_v1/RECO
                    /Cosmics/Commissioning08_CRUZET4_V2P_CRUZET4_InterimReco_v3/RECO
                    /Cosmics/Commissioning08-CRUZET4_v1/RECO
                    /Cosmics/Commissioning08-CRUZET4_v1/RAW
                    /Cosmics/Commissioning08-MW33_v1/RECO
                    /Cosmics/Commissioning08-MW33_v1/RAW
                    
     -o, --outpath PATH
         path to copy job output *.root files to;
         currently (almost) no check performed;
         must be in AFS or CASTOR
         default: /castor/cern.ch/user/c/cctrack/DQM
         
     -m, --mergepath PATH
         path to merge the job output *.root files;
         currently (almost) no check performed;
         must be in AFS
         default: /afs/cern.ch/cms/CAF/CMSCOMM/COMM_TRACKER/DQM/SiStrip/jobs/merged
"""                        
LSTR_true  = ['1','TRUE' ,'True' ,'true' ]
LSTR_false = ['0','FALSE','False','false']
LSTR_datatiers = ['RECO','RAW']
# argument vector
LSTR_wordArgument = sys.argv[1:]
# default arguments
STR_server     = 'None'
LSTR_server    = [STR_server,'caf','bari','lnl2']
STR_email      = 'volker.adler@cern.ch'
INT_nJobs      = 10
BOOL_filtersOn = False
STR_dataset    = '/Cosmics/Commissioning08-EW35_3T_v1/RECO'
DICT_datasets = { '/Cosmics/EW35_3T_v1_CRUZET4_V3P_SuperPointing_v1/RECO'           :'/store/data/EW35_3T_v1/Cosmics/RECO/CRUZET4_V3P_SuperPointing_v1'           ,
                  '/Cosmics/EW35_3T_v1_CRUZET4_V3P_TrackerPointing_v1/RECO'         :'/store/data/EW35_3T_v1/Cosmics/RECO/CRUZET4_V3P_TrackerPointing_v1'         ,
                  STR_dataset                                                       :'/store/data/Commissioning08/Cosmics/RECO/EW35_3T_v1'                        ,
                  '/Cosmics/Commissioning08-EW35_3T_v1/RAW'                         :'/store/data/Commissioning08/Cosmics/RAW/EW35_3T_v1'                         ,
                  '/Cosmics/CRUZET4_v1_CRZT210_V1_SuperPointing_v1/RECO'            :'/store/data/CRUZET4_v1/Cosmics/RECO/CRZT210_V1_SuperPointing_v1'            ,
                  '/Cosmics/CRUZET4_v1_CRZT210_V1_TrackerPointing_v1/RECO'          :'/store/data/CRUZET4_v1/Cosmics/RECO/CRZT210_V1_TrackerPointing_v1'          ,
                  '/Cosmics/Commissioning08_CRUZET4_V2P_CRUZET4_InterimReco_v3/RECO':'/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V2P_CRUZET4_InterimReco_v3',
                  '/Cosmics/Commissioning08-CRUZET4_v1/RECO'                        :'/store/data/Commissioning08/Cosmics/RECO/CRUZET4_v1'                        ,
                  '/Cosmics/Commissioning08-CRUZET4_v1/RAW'                         :'/store/data/Commissioning08/Cosmics/RAW/CRUZET4_v1'                         ,
                  '/Cosmics/Commissioning08-MW33_v1/RECO'                           :'/store/data/Commissioning08/Cosmics/RECO/MW33_v1'                           ,
                  '/Cosmics/Commissioning08-MW33_v1/RAW'                            :'/store/data/Commissioning08/Cosmics/RAW/MW33_v1'                            }
STR_pathOut    = '/castor/cern.ch/user/c/cctrack/DQM'
STR_pathMerge  = '/afs/cern.ch/cms/CAF/CMSCOMM/COMM_TRACKER/DQM/SiStrip/jobs/merged'
# option lists
LSTR_functionLetters = ['-s','-c','-h']
DICT_functionLetters = {'--submit':LSTR_functionLetters[0],
                        '--create':LSTR_functionLetters[1],
                        '--help'  :LSTR_functionLetters[2]}
LSTR_optionLetters   = ['-r','-C','-S','-e','-j','-f','-d','-o','-m']
DICT_optionLetters   = {'--run'      :LSTR_optionLetters[0],  
                        '--CRAB'     :LSTR_optionLetters[1],  
                        '--server'   :LSTR_optionLetters[2],  
                        '--email'    :LSTR_optionLetters[3],  
                        '--jobs'     :LSTR_optionLetters[4],
                        '--filter'   :LSTR_optionLetters[5],
                        '--dataset'  :LSTR_optionLetters[6],
                        '--outpath'  :LSTR_optionLetters[7],
                        '--mergepath':LSTR_optionLetters[8]}
                        
# Globals

global dict_arguments
dict_arguments = {}                                             
global str_runNumber
global bool_CRAB 
global str_server
global str_email
global int_nJobs     
global bool_filtersOn
global str_dataset   
global str_datatier   
global str_pathOut
global str_pathMerge

## Function Usage()
#
#  Displays usage of the script
def Usage():
  """ Function Usage():
  Displays usage of the script
  """
  print STR_textUsage
  
## Main program

print

# Check function letters

if len(LSTR_wordArgument) == 0:
  Usage()
  sys.exit(1)
for str_argument in LSTR_wordArgument:
  if str_argument in LSTR_functionLetters       or\
     DICT_functionLetters.has_key(str_argument)   :
    break
  else:
    Usage()
    sys.exit(1)
    
# Check options

str_argumentFormer = ''
bool_standBy       = False
for str_argument in LSTR_wordArgument:
  if not ( str_argument in LSTR_functionLetters or DICT_functionLetters.has_key(str_argument) or\
           str_argument in LSTR_optionLetters   or DICT_optionLetters.has_key(str_argument)     ):
    if str_argument[0] == '-':
      print '> submitDQMOfflineCAF_cfg.py > unknown option used'
      print '                               exit'
      print
      Usage()           
      sys.exit(1)
    if not bool_standBy:
      print '> submitDQMOfflineCAF_cfg.py > value without option used'
      print '                               exit'
      print
      Usage()
      sys.exit(1)
    dict_arguments[str_argumentFormer] = str_argument
    bool_standBy                       = False
  else:
    if bool_standBy:
      dict_arguments[str_argumentFormer] = STR_default
      if str_argumentFormer in LSTR_optionLetters or\
         DICT_optionLetters.has_key(str_argumentFormer):
        print '> submitDQMOfflineCAF_cfg.py > option "%s" w/o value' %(str_argumentFormer)
        print '                               default used'
        print
    bool_standBy = not ( str_argument in LSTR_functionLetters       or\
                         DICT_functionLetters.has_key(str_argument)   )
    if not bool_standBy:
      dict_arguments[str_argument] = STR_default
  str_argumentFormer = str_argument
if bool_standBy:
  dict_arguments[str_argumentFormer] = STR_default
  if str_argumentFormer in LSTR_optionLetters       or\
     DICT_optionLetters.has_key(str_argumentFormer)   :
    print '> submitDQMOfflineCAF_cfg.py > option "%s" w/o value' %(str_argumentFormer)
    print '                               default used'
    print
    
# Correct arguments' dictionary

dict_arguments2 = dict_arguments
for str_key, str_value in dict_arguments2.items():
  if str_key in DICT_functionLetters.keys():
    del dict_arguments[str_key]
    dict_arguments[DICT_functionLetters[str_key]] = str_value
  if str_key in DICT_optionLetters.keys():
    del dict_arguments[str_key]
    dict_arguments[DICT_optionLetters[str_key]] = str_value
    
# Help (exit)

if dict_arguments.has_key(LSTR_functionLetters[2]):
  Usage()
  sys.exit(0)
  
# Check and assign arguments

if dict_arguments.has_key(LSTR_optionLetters[0])        and\
   dict_arguments[LSTR_optionLetters[0]] != STR_default    :
  str_runNumber = dict_arguments[LSTR_optionLetters[0]]
else:   
  print '> submitDQMOfflineCAF_cfg.py > no run number given'
  print '                               exit'
  print
  sys.exit(1)
if dict_arguments.has_key(LSTR_optionLetters[1])        and\
   dict_arguments[LSTR_optionLetters[1]] != STR_default    :
  bool_CRAB = dict_arguments[LSTR_optionLetters[1]]
else:   
  bool_CRAB = True
  str_buffer  = commands.getoutput('which crab')
  if str_buffer.find('which: no crab in') >= 0:
    print '> submitDQMOfflineCAF_cfg.py > CRAB environment not set properly;'
    print '                               please use'
    print
    print '                               $ source /afs/cern.ch/cms/ccs/wm/scripts/Crab/crab.csh'
    print
    print '                               exit'
    print
    sys.exit(1)
if dict_arguments.has_key(LSTR_optionLetters[2])        and\
   dict_arguments[LSTR_optionLetters[2]] != STR_default    :
  str_server = dict_arguments[LSTR_optionLetters[2]]
  if not str_server in LSTR_server:
    print '> submitDQMOfflineCAF_cfg.py > CRAB server "%s" not available' %(str_server)
    print '                               exit'
    print
    sys.exit(1)
  elif str_server == LSTR_server[2]:                   
    print '> submitDQMOfflineCAF_cfg.py > CRAB server "%s" currently in drain mode' %(str_server)
    print '                               and not available for submission'
    print '                               exit'
    print
    sys.exit(1)
  # FIXME: CRAB server submission disabled at the moment.
  elif not str_server == STR_server: 
    print '> submitDQMOfflineCAF_cfg.py > CRAB server submission disabled at the moment'
    print '                               exit'
    print
    sys.exit(1)
else:   
  str_server = STR_server
if dict_arguments.has_key(LSTR_optionLetters[3])        and\
   dict_arguments[LSTR_optionLetters[3]] != STR_default    :
  str_email = dict_arguments[LSTR_optionLetters[3]]
else:   
  str_email = STR_email
if dict_arguments.has_key(LSTR_optionLetters[4])        and\
   dict_arguments[LSTR_optionLetters[4]] != STR_default    :
  int_nJobs = int(dict_arguments[LSTR_optionLetters[4]])
else:   
  int_nJobs = INT_nJobs
if dict_arguments.has_key(LSTR_optionLetters[5])        and\
   dict_arguments[LSTR_optionLetters[5]] != STR_default    :
  if dict_arguments[LSTR_optionLetters[5]] in LSTR_true:
    bool_filtersOn = True
  elif dict_arguments[LSTR_optionLetters[5]] in LSTR_false:  
    bool_filtersOn = False
  else:
    print '> submitDQMOfflineCAF_cfg.py > option \'-f\' expects 0/1, FALSE/TRUE, False/True or false/true'
    print '                               exit'
    print
    sys.exit(1)
else:   
  bool_filtersOn = BOOL_filtersOn
if dict_arguments.has_key(LSTR_optionLetters[6])        and\
   dict_arguments[LSTR_optionLetters[6]] != STR_default    :
  str_dataset = dict_arguments[LSTR_optionLetters[6]]
else:   
  str_dataset = STR_dataset
# FIXME: more sophisticated LFN determination for dataset
if not DICT_datasets.has_key(str_dataset):
  print '> submitDQMOfflineCAF_cfg.py > dataset "%s" not registered' %(str_dataset)
  print '                               exit'
  print
  sys.exit(1)
str_datatier = string.split(str_dataset, '/')[-1]
# FIXME: more sophisticated magn. field determination for dataset
str_magField = STR_magField0
if str_dataset.find('_3T_') >= 0:
  str_magField = STR_magField30
if not str_datatier in LSTR_datatiers:
  print '> submitDQMOfflineCAF_cfg.py > datatier "%s" not processable' %(str_datatier)
  print '                               exit'
  print
  sys.exit(1)
if dict_arguments.has_key(LSTR_optionLetters[7])        and\
   dict_arguments[LSTR_optionLetters[7]] != STR_default    :
  str_pathOut = dict_arguments[LSTR_optionLetters[7]]
else:   
  str_pathOut = STR_pathOut
bool_useCastor = True
if string.split(str_pathOut,'/')[1] == 'afs':
  bool_useCastor = False
elif string.split(str_pathOut,'/')[1] != 'castor':
  print '> submitDQMOfflineCAF_cfg.py > output path not accepted'
  print '                               exit'
  print
  Usage()
  sys.exit(1)
if dict_arguments.has_key(LSTR_optionLetters[8])        and\
   dict_arguments[LSTR_optionLetters[8]] != STR_default    :
  str_pathMerge = dict_arguments[LSTR_optionLetters[8]]
else:   
  str_pathMerge = STR_pathMerge
if string.split(str_pathMerge,'/')[1] != 'afs':
  print '> submitDQMOfflineCAF_cfg.py > merge path not accepted'
  print '                               exit'
  print
  Usage()
  sys.exit(1)
  
# Action
  
# current environment

str_pathCurrentDir        = os.getcwd()
str_pathCmsswBase         = os.getenv('CMSSW_BASE')
str_nameCmsswRel          = os.path.basename(str_pathCmsswBase)
str_pathCmsswBaseSrc      = str_pathCmsswBase + '/src'
str_pathCmsswBasePackage  = str_pathCmsswBaseSrc + '/' + STR_nameCmsswPackage

# prepare work area
  
int_nDigits = 1
if int(str_runNumber) >= 10:
  int_nDigits = int(math.log10(int(str_runNumber))) + 1
str_nameRun = 'R'
for int_iJob in range(9-int_nDigits):
  str_nameRun += '0'
str_nameRun += str_runNumber

if os.path.exists(str_nameRun):
  for str_root, str_dirs, str_files in os.walk(str_nameRun, topdown = False):
    for name in str_files:
      os.remove(os.path.join(str_root, name))
    for name in str_dirs:
      os.rmdir(os.path.join(str_root, name))
  os.rmdir(str_nameRun)
os.mkdir(str_nameRun)
str_pathRunIncludeDir = str_pathCmsswBasePackage + '/data/' + str_nameRun
if os.path.exists(str_pathRunIncludeDir):
  for str_root, str_dirs, str_files in os.walk(str_pathRunIncludeDir, topdown = False):
    for name in str_files:
      os.remove(os.path.join(str_root, name))
    for name in str_dirs:
      os.rmdir(os.path.join(str_root, name))
  os.rmdir(str_pathRunIncludeDir)
os.mkdir(str_pathRunIncludeDir)
str_nameInputFilesFile = str_nameRun + '/' + str_nameRun + '.txt'

# dealing with input files

int_nInputFiles    = 0
file_inputFilesCff = file(str_nameInputFilesFile, 'w')
str_dbsParams    = urllib.urlencode({'dbsInst': 'cms_dbs_prod_global', 'blockName': '*', 'dataset': str_dataset, 'userMode': 'user', 'run': str_runNumber, 'what': 'cff'})
lstr_dbsOutput   = urllib.urlopen("https://cmsweb.cern.ch/dbs_discovery/getLFN_txt", str_dbsParams)
str_pathDbsStore = DICT_datasets[str_dataset]
for str_iLine in lstr_dbsOutput.readlines():
  if str_iLine.find(str_pathDbsStore) > -1:
    int_nInputFiles += 1
    file_inputFilesCff.write(str_iLine)
if int_nInputFiles == 0:
  print '> submitDQMOfflineCAF_cfg.py > no input files found in DBS for run ' + str_runNumber
  print '                               exit'
  sys.exit(1)
file_inputFilesCff.close()
if int_nInputFiles <= int_nJobs:
  int_nJobs = int_nInputFiles
if int_nJobs == 0:
  int_nJobs = 1
nInputFilesJob = int(int_nInputFiles/int_nJobs) + 1
if int_nInputFiles%int_nJobs == 0:
  nInputFilesJob -= 1
if int_nInputFiles == nInputFilesJob*(int_nJobs-1) and int_nJobs > 1:
  int_nJobs -= 1
str_nJobs = str(int_nJobs)
print '> submitDQMOfflineCAF_cfg.py > input files for run ' + str_runNumber + ':   ' + str(int_nInputFiles)
print

# prepare scripts

# prepare merge script  
int_nLinesRead      = 0
file_inputFilesCff  = file(str_nameInputFilesFile, 'r')
lstr_linesInput     = file_inputFilesCff.readlines()
str_nameMergeScript = 'merge' + str_nameRun + '.job'
file_mergeScript = file(str_nameRun + '/' + str_nameMergeScript, 'w')
file_mergeScript.write('#!/bin/tcsh\n')
file_mergeScript.write('cd ' + str_pathCmsswBaseSrc + '\n')
file_mergeScript.write('cmsenv\n')
file_mergeScript.write('setenv STAGE_SVCCLASS cmscaf\n')
file_mergeScript.write('hadd -f ' + str_pathMerge + '/DQM_SiStrip_' + str_nameRun + '_CAF-' + str_nameCmsswRel +'-standAlone.root \\\n') # FIXME: make configurable
# create harvesting config file
str_sedCommand  = 'sed '
str_sedCommand += '-e \"s#xRUN_NUMBERx#'         + str_runNumber + '#g\" '
str_sedCommand += '-e \"s#xMERGED_INPUT_FILEx#'  + str_pathMerge + '/DQM_SiStrip_' + str_nameRun + '_CAF-' + str_nameCmsswRel +'-standAlone.root#g\" '
str_sedCommand += '-e \"s#xMERGED_OUTPUT_FILEx#' + str_pathMerge + '/DQM_SiStrip_' + str_nameRun + '_CAF-' + str_nameCmsswRel +'.root#g\" '
str_sedCommand += str_pathCmsswBasePackage + '/test/SiStripCAFHarvest_template.cfg > ' + str_nameRun + '/SiStripCAFHarvest.cfg'
os.system(str_sedCommand)

# loop over single jobs
if bool_CRAB:
  os.chdir(str_nameRun)     
  str_nameRunIncludeDir = STR_nameCmsswPackage + '/data/' + str_nameRun
  str_outputDir         = '.'
  # create main configuration file
  str_sedCommand = 'sed '
  if bool_filtersOn:
    str_sedCommand += '-e \"s#xHLT_FILTERx#    hltFilter,#g\" '
  else:
    str_sedCommand += '-e \"s#xHLT_FILTERx#//     hltFilter,#g\" '
  if str_datatier == 'RECO':
    str_sedCommand += '-e \"s#xRECO_FROM_RAWx#//     SiStripDQMRecoFromRaw,#g\" '
    str_sedCommand += '-e \"s#xDQM_FROM_RAWx#//     SiStripDQMSourceGlobalRunCAF_fromRAW,#g\" '
  else:
    str_sedCommand += '-e \"s#xRECO_FROM_RAWx#    SiStripDQMRecoFromRaw,#g\" '
    str_sedCommand += '-e \"s#xDQM_FROM_RAWx#    process.SiStripDQMSourceGlobalRunCAF_fromRAW,#g\" '
  str_sedCommand += '-e \"s#xMAG_FIELDx#'         + str_magField                                      + '#g\" '
  str_sedCommand += '-e \"s#xINCLUDE_DIRECTORYx#' + str_nameRunIncludeDir + '#g\" '
  str_sedCommand += '-e \"s#xINPUT_FILESx#'       + str_nameRunIncludeDir + '/' + STR_nameInputFilesJobCff + '#g\" '
  str_sedCommand += str_pathCmsswBasePackage + '/test/SiStripDQMOfflineGlobalRunCAF_template.cfg > SiStripDQMOfflineGlobalRunCAF.cfg'
  os.system(str_sedCommand)
  # create included input files list
  str_pathInputFilesJobCff = str_pathRunIncludeDir + '/' + STR_nameInputFilesJobCff
  file_inputFilesJobCff = file(str_pathInputFilesJobCff, 'w')
  file_inputFilesJobCff.write('  source = PoolSource {\n    untracked vstring fileNames = {\n')
  for str_linesInput in lstr_linesInput:
    file_inputFilesJobCff.write(str_linesInput)
  file_inputFilesJobCff.write('    }\n  }\n')
  file_inputFilesJobCff.close()
  # create included configuration file
  str_sedCommand = 'sed '
  str_sedCommand += '-e \"s#xOUTPUT_DIRECTORYx#' + str_outputDir + '#g\" '
  str_sedCommand += str_pathCmsswBasePackage + '/data/SiStripDQMOfflineGlobalRunCAF_template.cff > ' + str_pathRunIncludeDir + '/SiStripDQMOfflineGlobalRunCAF.cff'
  os.system(str_sedCommand)
  for int_iJob in range(int_nJobs):
    for n_iActualLine in range(int_nLinesRead, min(int_nLinesRead+nInputFilesJob, int_nInputFiles)):
      int_nLinesRead += 1
    # extend merge script
    str_nJobs = str(int_iJob+1)
    str_lineMergeScript = str_pathOut + '/DQM_SiStrip_' + str_nameRun + '_' + str_nJobs + '.root'
    if bool_useCastor:
      str_lineMergeScript = 'rfio:' + str_lineMergeScript
    if int_nLinesRead < int_nInputFiles:
      str_lineMergeScript += ' \\'
    str_lineMergeScript += '\n'  
    file_mergeScript.write(str_lineMergeScript)
    # FIXME: This protection is currently needed. Review calculations again!
    if int_nLinesRead >= int_nInputFiles:
      print '> submitDQMOfflineCAF_cfg.py > number of created job: ' + str_nJobs
      print
      break
  os.chdir(str_pathCurrentDir)
else:
  for int_iJob in range(int_nJobs):
    int_nDigits = 1
    if int_iJob >= 10:
      int_nDigits = int(math.log10(int_iJob)) + 1
    str_nameJob = str_nameRun + "_"
    for int_iDigit in range(4-int_nDigits):
      str_nameJob += '0'
    str_nameJob += str(int_iJob)
    # prepare job dir
    str_nameJobDir = str_nameRun + "/" + str_nameJob
    os.mkdir(str_nameJobDir)
    os.chdir(str_nameJobDir)     
    str_nameJobIncludeDir = STR_nameCmsswPackage + '/data/' + str_nameJobDir
    str_pathJobIncludeDir = str_pathRunIncludeDir + '/' + str_nameJob
    str_outputDir         = '/tmp/' + os.getenv('USER') + '/' + str_nameJobDir
    # create main configuration file
    str_sedCommand = 'sed '
    if bool_filtersOn:
      str_sedCommand += '-e \"s#xHLT_FILTERx#    hltFilter,#g\" '
    else:
      str_sedCommand += '-e \"s#xHLT_FILTERx#//     hltFilter,#g\" '
    if str_datatier == 'RECO':
      str_sedCommand += '-e \"s#xRECO_FROM_RAWx#//     SiStripDQMRecoFromRaw,#g\" '
      str_sedCommand += '-e \"s#xDQM_FROM_RAWx#//     SiStripDQMSourceGlobalRunCAF_fromRAW,#g\" '
    else:
      str_sedCommand += '-e \"s#xRECO_FROM_RAWx#    SiStripDQMRecoFromRaw,#g\" '
      str_sedCommand += '-e \"s#xDQM_FROM_RAWx#    SiStripDQMSourceGlobalRunCAF_fromRAW,#g\" '
    str_sedCommand += '-e \"s#xMAG_FIELDx#'         + str_magField                                      + '#g\" '
    str_sedCommand += '-e \"s#xINCLUDE_DIRECTORYx#' + str_pathJobIncludeDir                                  + '#g\" '
    str_sedCommand += '-e \"s#xINPUT_FILESx#'       + str_pathJobIncludeDir + '/' + STR_nameInputFilesJobCff + '#g\" '
    str_sedCommand += str_pathCmsswBasePackage + '/test/SiStripDQMOfflineGlobalRunCAF_template.cfg > SiStripDQMOfflineGlobalRunCAF.cfg'
    os.system(str_sedCommand)
    # prepare job include dir
    os.mkdir(str_pathJobIncludeDir)
    # create included input files list
    str_pathInputFilesJobCff = str_pathJobIncludeDir + '/' + STR_nameInputFilesJobCff
    file_inputFilesJobCff = file(str_pathInputFilesJobCff, 'w')
    file_inputFilesJobCff.write('  source = PoolSource {\n    untracked vstring fileNames = {\n')
    for n_iActualLine in range(int_nLinesRead, min(int_nLinesRead+nInputFilesJob, int_nInputFiles)):
      str_linesInput = lstr_linesInput[n_iActualLine]
      # fix commata and end of line
      str_actualLine = str_linesInput
      if (n_iActualLine+1)%nInputFilesJob == 0 or int_nLinesRead == int_nInputFiles-1:
        str_actualLine = string.split(str_linesInput, ',')[0] + '\n'
      file_inputFilesJobCff.write(str_actualLine)
      int_nLinesRead += 1
    file_inputFilesJobCff.write('    }\n  }\n')
    file_inputFilesJobCff.close()
    # extend merge script
    str_lineMergeScript = str_pathOut + '/DQM_SiStrip_' + str_nameJob + '.root'
    if bool_useCastor:
      str_lineMergeScript = 'rfio:' + str_lineMergeScript
    if int_nLinesRead < int_nInputFiles:
      str_lineMergeScript += ' \\'
    str_lineMergeScript += '\n'  
    file_mergeScript.write(str_lineMergeScript)
    # create included configuration file
    str_sedCommand = 'sed '
    str_sedCommand += '-e \"s#xOUTPUT_DIRECTORYx#' + str_outputDir + '#g\" '
    str_sedCommand += str_pathCmsswBasePackage + '/data/SiStripDQMOfflineGlobalRunCAF_template.cff > ' + str_pathJobIncludeDir + '/SiStripDQMOfflineGlobalRunCAF.cff'
    os.system(str_sedCommand)
    # create job script
    str_sedCommand = 'sed '
    str_sedCommand += '-e \"s#xCMSSW_BASEx#'  + str_pathCmsswBase  + '#g\" '
    str_sedCommand += '-e \"s#xRUN_NAMEx#'    + str_nameRun        + '#g\" '
    str_sedCommand += '-e \"s#xJOB_NAMEx#'    + str_nameJob        + '#g\" '
    str_sedCommand += '-e \"s#xCURRENT_DIRx#' + str_pathCurrentDir + '#g\" '
    str_sedCommand += '-e \"s#xSUFFIXx#'      + '.cfg'             + '#g\" '
    if bool_useCastor:
      str_sedCommand += '-e \"s#xCOPYx#rfcp#g\" '
    else:
      str_sedCommand += '-e \"s#xCOPYx#cp#g\" '
    str_sedCommand += '-e \"s#xOUTPUT_DIRx#' + str_pathOut + '#g\" '
    str_sedCommand += str_pathCmsswBasePackage + '/scripts/SiStripDQMOfflineCAF_template.job > SiStripDQMOfflineCAF.job'
    os.system(str_sedCommand)
    # finalize job creation
    os.chdir(str_pathCurrentDir)
    # FIXME: This protection is currently needed. Review calculations again!
    if int_nLinesRead >= int_nInputFiles:
      str_nJobs = str(int_iJob+1)
      print '> submitDQMOfflineCAF_cfg.py > number of created jobs: ' + str_nJobs
      print
      break

# compile

os.chdir(str_pathCmsswBasePackage)
os.system('scramv1 b python')
os.chdir(str_pathCurrentDir)
print
    
# finish scripts
    
# finish merge script
file_mergeScript.write('cd ' + str_pathCurrentDir + '/' + str_nameRun + '/\n')
file_mergeScript.write('cmsRun SiStripCAFHarvest.cfg\n')
file_mergeScript.close()
# create CRAB configuration
if bool_CRAB:
  os.chdir(str_nameRun)     
  str_sedCommand  = 'sed '
  str_sedCommand += '-e \"s#xSERVER_NAMEx#'        + str_server    + '#g\" '
  str_sedCommand += '-e \"s#xDATASETPATHx#'        + str_dataset   + '#g\" '
  str_sedCommand += '-e \"s#xRUNSELECTIONx#'       + str_runNumber + '#g\" '
  str_sedCommand += '-e \"s#xNUMBER_OF_JOBSx#'     + str_nJobs     + '#g\" '
  str_sedCommand += '-e \"s#xEMAILx#'              + str_email     + '#g\" '
  str_sedCommand += '-e \"s#xOUTPUT_FILEx#'        + str_nameRun   + '#g\" '
  str_sedCommand += '-e \"s#xUI_WORKING_DIRx#crab' + str_nameRun   + '#g\" '
  str_sedCommand += '-e \"s#xSTORAGE_PATHx#'       + str_pathOut   + '#g\" '
  str_sedCommand += '-e \"s#xSUFFIXx#'             + '.cfg'        + '#g\" '
  if bool_useCastor:
    str_sedCommand += '-e \"s#xCOPY_DATAx#1#g\" '
  else:
    str_sedCommand += '-e \"s#xCOPY_DATAx#0#g\" '
  str_sedCommand += str_pathCmsswBasePackage + '/test/SiStripDQMOfflineCAF_template.crab > crab.cfg'
  os.system(str_sedCommand)
  os.system('crab -create')
  os.chdir(str_pathCurrentDir)

# submit jobs

if dict_arguments.has_key(LSTR_functionLetters[0]):
  if bool_CRAB:
    os.chdir(str_nameRun)
    print '> submitDQMOfflineCAF.cfg >'
    print '  ' + os.getcwd() + ' : crab -submit -c crab' + str_nameRun
    os.system('crab -submit -c crab' + str_nameRun)
    print
    os.chdir(str_pathCurrentDir)
    time.sleep(5)
    os.system('crab -status -c ' + str_nameRun + '/crab' + str_nameRun)
  else:
    for int_iJob in range(int_nJobs):
      int_nDigits = 1
      if int_iJob >= 10:
        int_nDigits = int(math.log10(int_iJob)) + 1
      str_nameJobDir = str_nameRun + "/" + str_nameRun + "_"
      for int_iDigit in range(4-int_nDigits):
        str_nameJobDir += '0'
      str_nameJobDir += str(int_iJob)
      os.chdir(str_nameJobDir)     
      os.chmod('SiStripDQMOfflineCAF.job',OCT_rwx_r_r)
      print '> submitDQMOfflineCAF.cfg >'
      print '  ' + os.getcwd() + ' : bsub -q cmscaf SiStripDQMOfflineCAF.job'
      os.system('bsub -q cmscaf SiStripDQMOfflineCAF.job')
      print
      os.chdir(str_pathCurrentDir)
    time.sleep(5)
    os.system('bjobs -q cmscaf')
  os.chmod(str_nameRun + '/' + str_nameMergeScript,OCT_rwx_r_r)
