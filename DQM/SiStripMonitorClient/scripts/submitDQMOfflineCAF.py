#!/usr/bin/env python

#
# $Id: submitDQMOfflineCAF.py,v 1.1 2008/08/02 16:06:09 dutta Exp $
#

## CMSSW/DQM/SiStripMonitorClient/scripts/submitDQMOfflineCAF.py
#
#  This script submits batch jobs to the CAF in order to process the full
#  granularity SiStrip offline DQM.
#  Questions and comments to: volker.adler@crn.ch


import sys
import os
import os.path
import shutil
import string
import math
import urllib
import time

# Constants

# argument vector
LSTR_wordArgument = sys.argv[1:]
# default arguments
INT_nJobs      = 10
BOOL_filtersOn = False
STR_dataset    = '/Cosmics/CRUZET3_CRUZET3_V2P_v3/RECO'
STR_pathOut    = os.getenv('CASTOR_HOME') + '/DQM'
STR_pathMerge  = os.getenv('HOME') + '/scratch0/DQM'
# numbers
OCT_rwx_r_r = 0744
# strings
STR_default   = 'DEFAULT'
STR_textUsage = """ CMSSW/DQM/SiStripMonitorClient/scripts/submitDQMOfflineCAF.py
 
 This script submits batch jobs to the CAF in order to process the full
 granularity SiStrip offline DQM.
 Questions and comments to: volker.adler@cern.ch
 
 Usage(): submitDQMOfflineCAF.py (-s, --submit | -c, --create |
                                  -h, --help)
                                 [-r, --run]
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
         
     -j, --jobs NUMBER
         number of jobs to create;
         default: 10
         
     -f, --filter TRUE/FALSE
         use or use not HLT filters to select events to process;
         default: False
      
     -d, --dataset PRIMARY_DATASET
         specify dataset for DBS query;
         default: /Cosmics/CRUZET3_CRUZET3_V2P_v3/RECO
         
     -o, --outpath PATH
         path to copy job output *.root files to;
         currently (almost) no check performed;
         must be in AFS or CASTOR
         default: $CASTOR_HOME/DQM
         
     -m, --mergepath PATH
         path to merge the job output *.root files;
         currently (almost) no check performed;
         must be in AFS
         default: $HOME/scratch0/DQM
"""                        
LSTR_true  = ['1','TRUE' ,'True' ,'true' ]
LSTR_false = ['0','FALSE','False','false']
DICT_datasets = {STR_dataset:'/store/data/CRUZET3/Cosmics/RECO/CRUZET3_V2P_v3'}
# option lists
LSTR_functionLetters = ['-s','-c','-h']
DICT_functionLetters = {'--submit':LSTR_functionLetters[0],
                        '--create':LSTR_functionLetters[1],
                        '--help'  :LSTR_functionLetters[2]}
LSTR_optionLetters   = ['-r','-j','-f','-d','-o','-m']
DICT_optionLetters   = {'--run'      :LSTR_optionLetters[0],  
                        '--jobs'     :LSTR_optionLetters[1],
                        '--filter'   :LSTR_optionLetters[2],
                        '--dataset'  :LSTR_optionLetters[3],
                        '--outpath'  :LSTR_optionLetters[4],
                        '--mergepath':LSTR_optionLetters[5]}
                        
# Globals

global dict_arguments
dict_arguments = {}                                             
global str_runNumber 
global int_nJobs     
global bool_filtersOn
global str_dataset   
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
      print '> submitDQMOfflineCAF.py > unknown option used'
      print '                           exit'
      print
      Usage()           
      sys.exit(1)
    if not bool_standBy:
      print '> submitDQMOfflineCAF.py > value without option used'
      print '                           exit'
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
        print '> submitDQMOfflineCAF.py > option "%s" w/o value' %(str_argumentFormer)
        print '                           default used'
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
    print '> submitDQMOfflineCAF.py > option "%s" w/o value' %(str_argumentFormer)
    print '                           default used'
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
  print '> submitDQMOfflineCAF.py > no run number given'
  print '                           exit'
  print
  sys.exit(1)
if dict_arguments.has_key(LSTR_optionLetters[1])        and\
   dict_arguments[LSTR_optionLetters[1]] != STR_default    :
  int_nJobs = int(dict_arguments[LSTR_optionLetters[1]])
else:   
  int_nJobs = INT_nJobs
if dict_arguments.has_key(LSTR_optionLetters[2])        and\
   dict_arguments[LSTR_optionLetters[2]] != STR_default    :
  if dict_arguments[LSTR_optionLetters[2]] in LSTR_true:
    bool_filtersOn = True
  elif dict_arguments[LSTR_optionLetters[2]] in LSTR_false:  
    bool_filtersOn = False
  else:
    print '> submitDQMOfflineCAF.py > option \'-f\' expects 0/1, FALSE/TRUE, False/True or false/true'
    print '                           exit'
    print
    sys.exit(1)
else:   
  bool_filtersOn = BOOL_filtersOn
if dict_arguments.has_key(LSTR_optionLetters[3])        and\
   dict_arguments[LSTR_optionLetters[3]] != STR_default    :
  str_dataset = dict_arguments[LSTR_optionLetters[3]]
else:   
  str_dataset = STR_dataset
if not DICT_datasets.has_key(str_dataset):
  print '> submitDQMOfflineCAF.py > dataset "%s" not registered' %(str_dataset)
  print '                           exit'
  print
  sys.exit(1)
if dict_arguments.has_key(LSTR_optionLetters[4])        and\
   dict_arguments[LSTR_optionLetters[4]] != STR_default    :
  str_pathOut = dict_arguments[LSTR_optionLetters[4]]
else:   
  str_pathOut = STR_pathOut
bool_useCastor = True
if string.split(str_pathOut,'/')[1] == 'afs':
  bool_useCastor = False
elif string.split(str_pathOut,'/')[1] != 'castor':
  print '> submitDQMOfflineCAF.py > output path not accepted'
  print '                           exit'
  print
  Usage()
  sys.exit(1)
if dict_arguments.has_key(LSTR_optionLetters[5])        and\
   dict_arguments[LSTR_optionLetters[5]] != STR_default    :
  str_pathMerge = dict_arguments[LSTR_optionLetters[5]]
else:   
  str_pathMerge = STR_pathMerge
if string.split(str_pathMerge,'/')[1] != 'afs':
  print '> submitDQMOfflineCAF.py > merge path not accepted'
  print '                           exit'
  print
  Usage()
  sys.exit(1)
  
# Action
  
# current environment

# FIXME: This should be done in a more flexible way. See r1.4 for proposal...
str_pathCurrentDir        = os.getcwd()
str_pathCurrentDirStrings = string.split(str_pathCurrentDir, '/src/') # assumes to be in _the_ release working area
str_pathCmsswBase         = str_pathCurrentDirStrings[0]              # assumes to be in _the_ release working area 

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

# dealing with input files

int_nInputFiles       = 0
file_inputFilesCff = file(str_nameRun + '/' + str_nameRun + '.cff','w')
str_dbsParams    = urllib.urlencode({'dbsInst': 'cms_dbs_prod_global', 'blockName': '*', 'dataset': str_dataset, 'userMode': 'user', 'run': str_runNumber, 'what': 'cff'})
lstr_dbsOutput    = urllib.urlopen("https://cmsweb.cern.ch/dbs_discovery/getLFN_txt", str_dbsParams)
str_pathDbsStore = DICT_datasets[str_dataset]
for str_iLine in lstr_dbsOutput.readlines():
  if str_iLine.find(str_pathDbsStore) > -1:
    int_nInputFiles += 1
    file_inputFilesCff.write(str_iLine)
if int_nInputFiles == 0:
  print '> submitDQMOfflineCAF.py > no input files found in DBS for run ' + str_runNumber
  print '                           exit'
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
print '> submitDQMOfflineCAF.py > input files for run ' + str_runNumber + ':   ' + str(int_nInputFiles)
print

# loop over single jobs
  
int_nLinesRead = 0
file_inputFilesCff = file(str_nameRun + '/' + str_nameRun + '.cff', 'r')
lstr_linesInput = file_inputFilesCff.readlines()
str_nameMergeScript = 'merge' + str_nameRun + '.job'
file_mergeScript = file(str_nameRun + '/' + str_nameMergeScript, 'w')
file_mergeScript.write('hadd -f ' + str_pathMerge + '/DQM_SiStrip_' + str_nameRun + '_CAF-standAlone.root \\\n') # --> configurable
for int_iJob in range(int_nJobs):
  int_nDigits = 1
  if int_iJob >= 10:
    int_nDigits = int(math.log10(int_iJob)) + 1
  str_nameJob = str_nameRun + "_"
  for int_iDigit in range(4-int_nDigits):
    str_nameJob += '0'
  str_nameJob += str(int_iJob)
  str_pathJobDir   = str_nameRun + "/" + str_nameJob
  os.mkdir(str_pathJobDir)
  os.chdir(str_pathJobDir)
  str_nameInputFilesJobCff = 'inputFiles.cff'
  if bool_filtersOn:
    os.system('sed -e \"s#HLT_FILTER#    hltFilter,#g\" -e \"s#INCLUDE_DIRECTORY#' + str_pathCurrentDirStrings[1] + '/' + str_pathJobDir + '#g\" -e \"s#INPUT_FILES#' + str_pathCurrentDirStrings[1] + '/' + str_pathJobDir + '/' + str_nameInputFilesJobCff + '#g\" ' + str_pathCmsswBase + '/src/DQM/SiStripMonitorClient/test/SiStripDQMOfflineGlobalRunCAF_template.cfg > SiStripDQMOfflineGlobalRunCAF.cfg')
  else:
    os.system('sed -e \"s#HLT_FILTER#//     hltFilter,#g\" -e \"s#INCLUDE_DIRECTORY#' + str_pathCurrentDirStrings[1] + '/' + str_pathJobDir + '#g\" -e \"s#INPUT_FILES#' + str_pathCurrentDirStrings[1] + '/' + str_pathJobDir + '/' + str_nameInputFilesJobCff + '#g\" ' + str_pathCmsswBase + '/src/DQM/SiStripMonitorClient/test/SiStripDQMOfflineGlobalRunCAF_template.cfg > SiStripDQMOfflineGlobalRunCAF.cfg')
  file_inputFilesJobCff = file(str_nameInputFilesJobCff, 'w')
  file_inputFilesJobCff.write('  source = PoolSource {\n    untracked vstring fileNames = {\n')
  for n_iActualLine in range(int_nLinesRead, min(int_nLinesRead+nInputFilesJob, int_nInputFiles)):
    lstr_actualLine1  = string.split('      \'/store/data/' + string.split(lstr_linesInput[n_iActualLine], '/store/data/')[1], '.root\',')[0] + '.root\',\n'
    lstr_actualLine2  = lstr_actualLine1
    if (n_iActualLine+1)%nInputFilesJob == 0 or int_nLinesRead == int_nInputFiles-1:
      lstr_actualLine2 = string.split(lstr_actualLine1, ',')[0] + '\n'
    file_inputFilesJobCff.write(lstr_actualLine2)
    int_nLinesRead += 1
  file_inputFilesJobCff.write('    }\n  }\n')
  file_inputFilesJobCff.close()
  str_lineMergeScript = str_pathOut + '/DQM_SiStrip_' + str_nameJob + '.root'
  if bool_useCastor:
    str_lineMergeScript = 'rfio:' + str_lineMergeScript
  if int_nLinesRead < int_nInputFiles:
    str_lineMergeScript += ' \\'
  str_lineMergeScript += '\n'  
  file_mergeScript.write(str_lineMergeScript)
  os.system('sed -e \"s#OUTPUT_DIRECTORY#/tmp/' + os.getenv('USER') + '/' + str_pathJobDir + '#g\" ' + str_pathCmsswBase + '/src/DQM/SiStripMonitorClient/data/SiStripDQMOfflineGlobalRunCAF_template.cff > SiStripDQMOfflineGlobalRunCAF.cff')
  if bool_useCastor:
    os.system('sed -e \"s#CMSSW_BASE#' + str_pathCmsswBase + '#g\" -e \"s#RUN_NAME#' + str_nameRun + '#g\" -e \"s#JOB_NAME#' + str_nameJob + '#g\" -e \"s#CURRENT_DIR#' + str_pathCurrentDir + '#g\" -e \"s#COPY#rfcp#g\" -e \"s#OUTPUT_DIR#' + str_pathOut + '#g\" ' + str_pathCmsswBase + '/src/DQM/SiStripMonitorClient/scripts/SiStripDQMOfflineCAF_template.job > SiStripDQMOfflineCAF.job')
  else:
    os.system('sed -e \"s#CMSSW_BASE#' + str_pathCmsswBase + '#g\" -e \"s#RUN_NAME#' + str_nameRun + '#g\" -e \"s#JOB_NAME#' + str_nameJob + '#g\" -e \"s#CURRENT_DIR#' + str_pathCurrentDir + '#g\" -e \"s#COPY#cp#g\" -e \"s#OUTPUT_DIR#' + str_pathOut + '#g\" ' + str_pathCmsswBase + '/src/DQM/SiStripMonitorClient/scripts/SiStripDQMOfflineCAF_template.job > SiStripDQMOfflineCAF.job')
  os.chmod('SiStripDQMOfflineCAF.job',OCT_rwx_r_r)
  if dict_arguments.has_key(LSTR_functionLetters[0]):
    print '> submitDQMOfflineCAF.py >'
    print '  ' + os.getcwd() + ' : bsub -q cmscaf SiStripDQMOfflineCAF.job'
    print
    os.system('bsub -q cmscaf SiStripDQMOfflineCAF.job')
  os.chdir(str_pathCurrentDir)
  # FIXME: This protection is currently needed. Review calculations again!
  if int_nLinesRead >= int_nInputFiles:
    print '> submitDQMOfflineCAF.py > number of created job: ' + str(int_iJob+1)
    print
    break
file_mergeScript.close()
os.chmod(str_nameRun + '/' + str_nameMergeScript,OCT_rwx_r_r)

# check queue

if dict_arguments.has_key(LSTR_functionLetters[0]):
  time.sleep(5)
  os.system('bjobs -q cmscaf')
