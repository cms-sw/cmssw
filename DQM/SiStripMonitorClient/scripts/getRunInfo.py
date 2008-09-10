#!/usr/bin/env python

#
# $Id$
#

## CMSSW/DQM/SiStripMonitorClient/scripts/getRunInfo.py
#
#  For a given run, this script collects information useful for SiStrip DQM
#  from web sources.
#  Questions and comments to: volker.adler@cern.ch


import sys
import os
import string
import urllib
import datetime

# Constants

LSTR_arguments = sys.argv[1:]
# numbers
TD_shiftUTC = datetime.timedelta(hours = 2) # positive for timezones with later time than UTC
# strings
STR_SiStrip             = 'SiStrip'
STR_good                = 'SS_GOOD'
STR_htmlL1Key           = '&lt;b>L1&amp;nbsp;Key:&lt;/b>'
STR_htmlHLTKey          = '&lt;b>HLT&amp;nbsp;Key:&lt;/b>'
STR_wwwDBSData          = 'https://cmsweb.cern.ch/dbs_discovery/getData'
STR_headDatasets        = 'available data files'
LSTR_summaryKeys        = ['BField', 'HLT Version', 'L1 Rate', 'HLT Rate', 'L1 Triggers', 'HLT Triggers', 'LHC Fill', 'LHC Energy', 'Initial Lumi', 'Ending Lumi', 'Run Lumi', 'Run Live Lumi']
LSTR_summaryKeysTrigger = ['L1 Key', 'HLT Key']   
# STR_summaryHLTKey = 'HLT Key'</TH><TD><A HREF=HLTConfiguration?KEY=1551>/cdaq/cosmic/CRUZET3/HLTstartup_DTDataIntegrity/V1</A></TD></TR>

# Globals

global Str_run
global Dict_cmsmonRunRegistry
global Dict_cmsmonSubsystems
global Dict_cmsmonRunSummary
global Dict_dbsDatasets
global Lstr_hltPaths
# initialise
Str_run                = sys.argv[1]
Dict_cmsmonRunRegistry = {}
Dict_cmsmonSubsystems  = {}
Dict_cmsmonRunSummary  = {}
Dict_dbsDatasets       = {}
Lstr_hltPaths          = []

## FUNCTIONS

## Function Func_Exit()
#
#  Exit after error
def Func_Exit():
  """  Function Func_Exit():
  Exit after error
  """
  print '                  exit'
  print
  sys.exit(1)

## Func_GetHtmlTags(str_text)
#
# Gets HTML tags from a string
def Func_GetHtmlTags(str_text):
  """  Func_GetHtmlTags(str_text):
  Gets HTML tags from a string
  """
  lstr_split = str_text.split('</')
  lstr_tags  = []
  for str_split in lstr_split[1:]:
    lstr_tags.append(str_split.split('>')[0])
  return lstr_tags
 
## Func_GetHtmlTagValue(str_tag, str_text)
#
# Gets the value of a given HTML tag from a string
def Func_GetHtmlTagValue(str_tag, str_text):
  """  Func_GetHtmlTagValue(str_tag, str_text):
  Gets the value of a given HTML tag from a string
  """
  return str_text.split('<'+str_tag+'>')[1].split('</'+str_tag+'>')[0]

## Func_GetHtmlTagValues(str_text)
#
# Gets HTML tag values from a string
def Func_GetHtmlTagValues(str_text):
  """  Func_GetHtmlTagValues(str_text):
  Gets HTML tag values from a string
  """
  lstr_split   = str_text.split('</')
  lstr_values  = []
  for str_split in lstr_split[:-1]:
    lstr_values.append(str_split.split('>')[-1])
  return lstr_values
 
## Func_GetHtmlTagValueAttr(str_tag, str_text)
#
# Gets the attributes of a given HTML tag value from a string
def Func_GetHtmlTagValueAttr(str_value, str_text):
  """  Func_GetHtmlTagValueAttr(str_tag, str_text):
  Gets the attributes of a given HTML tag value from a string
  """
  return str_text.split('\">'+str_value+'<')[0].split('=\"')[-1]
 
## MAIN PROGRAM

print
print '> getRunInfo.py > information on run \t*** ' + Str_run + ' ***'
print

# Get run information from the web

# Run registry
# get run registry
str_cmsmonRunRegistry      = urllib.urlencode({'template':'text'})
file_cmsmonRunRegistry     = urllib.urlopen("http://cmsmon.cern.ch/runregistry/allrundata")
lstr_cmsmonRunRegistry     = []
str_cmsmonRunRegistryLong1 = ''
str_cmsmonRunRegistryLong2 = ''
for str_cmsmonRunRegistry in file_cmsmonRunRegistry.readlines():
  lstr_cmsmonRunRegistry.append(str_cmsmonRunRegistry) # store run registry information
  str_cmsmonRunRegistryLong1 += str_cmsmonRunRegistry
for str_cmsmonRunRegistry in str_cmsmonRunRegistryLong1.splitlines():
  str_cmsmonRunRegistryLong2 += str_cmsmonRunRegistry
if str_cmsmonRunRegistryLong2.find('<run id=\"'+Str_run+'\">') < 0:
  print '> getRunInfo.py > run ' + Str_run + ' not found in run registry'
  Func_Exit()
str_cmsmonRun = str_cmsmonRunRegistryLong2.split('<run id=\"'+Str_run+'\">')[1].split('</run>')[0]
for str_cmsmonHtmlTag in Func_GetHtmlTags(str_cmsmonRun):
  Dict_cmsmonRunRegistry[str_cmsmonHtmlTag] = Func_GetHtmlTagValue(str_cmsmonHtmlTag, str_cmsmonRun)
# check SiStrip
if Dict_cmsmonRunRegistry['subsystems'].find(STR_SiStrip) < 0:
  print '> getRunInfo.py > SiStrip was not in this run'
  Func_Exit()
str_htmlSubsystems = Dict_cmsmonRunRegistry['formatted_subsystems'].replace('&lt;','<')
for str_htmlSubsystem in Func_GetHtmlTagValues(str_htmlSubsystems):
  Dict_cmsmonSubsystems[str_htmlSubsystem] = Func_GetHtmlTagValueAttr(str_htmlSubsystem, str_htmlSubsystems)
print '> getRunInfo.py > SiStrip DQM from global : ' + Dict_cmsmonSubsystems[STR_SiStrip][3:]
if not Dict_cmsmonSubsystems[STR_SiStrip] == STR_good:
  Func_Exit()
print

# get run DBS entries
str_dbsRuns      = urllib.urlencode({'ajax':'0', '_idx':'0', 'pagerStep':'0', 'userMode':'user', 'release':'Any', 'tier':'Any', 'dbsInst':'cms_dbs_prod_global', 'primType':'Any', 'primD':'Any', 'minRun':Str_run, 'maxRun':Str_run})
file_dbsRuns     = urllib.urlopen("https://cmsweb.cern.ch/dbs_discovery/getRunsFromRange", str_dbsRuns)
lstr_dbsRuns     = []
lstr_dbsDatasets = []
for str_dbsRuns in file_dbsRuns.readlines():
  lstr_dbsRuns.append(str_dbsRuns) # store run DBS information
  if str_dbsRuns.find(STR_wwwDBSData) >= 0:
    if str_dbsRuns.split('&amp;proc=')[1].find('&amp;') >= 0:
      lstr_dbsDatasets.append(str_dbsRuns.split('&amp;proc=')[1].split('&amp;')[0])
    else:
      lstr_dbsDatasets.append(str_dbsRuns.split('&amp;proc=')[1])
int_maxLenDbsDatasets = 0
for str_dbsDatasets in lstr_dbsDatasets:
  str_dbsLFN  = urllib.urlencode({'dbsInst':'cms_dbs_prod_global', 'blockName':'*', 'dataset':str_dbsDatasets, 'userMode':'user', 'run':Str_run})
  file_dbsLFN = urllib.urlopen("https://cmsweb.cern.ch/dbs_discovery/getLFNlist", str_dbsLFN)
  for str_dbsLFN in file_dbsLFN.readlines():
    if str_dbsLFN.find('contians') >= 0 and str_dbsLFN.find('file(s)'):
      Dict_dbsDatasets[str_dbsDatasets] = str_dbsLFN.split()[1]
  if len(str_dbsDatasets) > int_maxLenDbsDatasets:
    int_maxLenDbsDatasets = len(str_dbsDatasets)
    
      
# get run summary
str_cmsmonRunSummary  = urllib.urlencode({'RUN':Str_run})
file_cmsmonRunSummary = urllib.urlopen("http://cmsmon.cern.ch/cmsdb/servlet/RunSummary", str_cmsmonRunSummary)
lstr_cmsmonRunSummary = []
for str_cmsmonRunSummary in file_cmsmonRunSummary.readlines():
  lstr_cmsmonRunSummary.append(str_cmsmonRunSummary) # store run summary information
  for str_summaryKeys in LSTR_summaryKeys:
    if str_cmsmonRunSummary.find(str_summaryKeys) >= 0:
      Dict_cmsmonRunSummary[str_summaryKeys] = str_cmsmonRunSummary.split('</TD></TR>')[0].split('>')[-1]
      break
  for str_summaryKeysTrigger in LSTR_summaryKeysTrigger:
    if str_cmsmonRunSummary.find(str_summaryKeysTrigger) >= 0:
      Dict_cmsmonRunSummary[str_summaryKeysTrigger] = str_cmsmonRunSummary.split('</A></TD></TR>')[0].split('>')[-1]
      if str_summaryKeysTrigger == 'HLT Key':
         Dict_cmsmonRunSummary['HLT Config ID'] = str_cmsmonRunSummary.split('HLTConfiguration?KEY=')[1].split('>')[0]
      break

# Determine further information

# get L1 and HLT key from run registry
str_htmlL1Key  = ''
str_htmlHLTKey = ''
if Dict_cmsmonRunRegistry.has_key('comment'):
  str_htmlL1Key  = Dict_cmsmonRunRegistry['comment'].split(STR_htmlL1Key)[1].split('&lt;')[0]
  str_htmlHLTKey = Dict_cmsmonRunRegistry['comment'].split(STR_htmlHLTKey)[1].split('&lt;')[0]
# get magnetic field
float_avMagMeasure = -999.0
dt_newStart        = datetime.datetime(2000,1,1,0,0,0)
dt_newEnd          = datetime.datetime(2000,1,1,0,0,0)
if ( Dict_cmsmonRunRegistry.has_key('RUN_START_TIME') and Dict_cmsmonRunRegistry.has_key('RUN_END_TIME') ):
  lstr_dateStart = Dict_cmsmonRunRegistry['RUN_START_TIME'].split(' ')[0].split('.')
  lstr_timeStart = Dict_cmsmonRunRegistry['RUN_START_TIME'].split(' ')[1].split(':')
  lstr_dateEnd   = Dict_cmsmonRunRegistry['RUN_END_TIME'].split(' ')[0].split('.')
  lstr_timeEnd   = Dict_cmsmonRunRegistry['RUN_END_TIME'].split(' ')[1].split(':')
  dt_oldStart    = datetime.datetime(int(lstr_dateStart[0]),int(lstr_dateStart[1]),int(lstr_dateStart[2]),int(lstr_timeStart[0]),int(lstr_timeStart[1]),int(lstr_timeStart[2]))
  dt_oldEnd      = datetime.datetime(int(lstr_dateEnd[0]),  int(lstr_dateEnd[1]),  int(lstr_dateEnd[2]),  int(lstr_timeEnd[0]),  int(lstr_timeEnd[1]),  int(lstr_timeEnd[2]))
  dt_newStart    = dt_oldStart - TD_shiftUTC
  dt_newEnd      = dt_oldEnd   - TD_shiftUTC
  str_cmsmonMagnetHistory  = urllib.urlencode({'TIME_BEGIN':dt_newStart, 'TIME_END':dt_newEnd})
  file_cmsmonMagnetHistory = urllib.urlopen("http://cmsmon.cern.ch/cmsdb/servlet/MagnetHistory", str_cmsmonMagnetHistory)
  float_avMagMeasure = -999.0
  for str_cmsmonMagnetHistory in file_cmsmonMagnetHistory.readlines():
    if str_cmsmonMagnetHistory.find('BFIELD, Tesla') >= 0:
      float_avMagMeasure = float(str_cmsmonMagnetHistory.split('</A>')[0].split('>')[-1])
else:
  print '> getRunInfo.py > cannot determine magnetic field due to missing time information' 
# get HLT configuration
str_cmsmonHLTConfig  = urllib.urlencode({'KEY':Dict_cmsmonRunSummary['HLT Config ID']})
file_cmsmonHLTConfig = urllib.urlopen("http://cmsmon.cern.ch/cmsdb/servlet/HLTConfiguration", str_cmsmonHLTConfig)
lstr_cmsmonHLTConfig = []
bool_foundPaths      = False
for str_cmsmonHLTConfig in file_cmsmonHLTConfig.readlines():
  lstr_cmsmonHLTConfig.append(str_cmsmonHLTConfig)
  if str_cmsmonHLTConfig.find('<H3>Paths</H3>') >= 0:
    bool_foundPaths = True
  if bool_foundPaths and str_cmsmonHLTConfig.find('<HR><H3>') >= 0:
    bool_foundPaths = False
  if bool_foundPaths and str_cmsmonHLTConfig.startswith('<TR><TD ALIGN=RIGHT>'):
    Lstr_hltPaths.append(str_cmsmonHLTConfig.split('</TD>')[1].split('<TD>')[-1])
    
    
# Print information

# from run registry
print '> getRunInfo.py > * information from run registry *'
print
if Dict_cmsmonRunRegistry.has_key('RUN_GLOBALNAME'):
  print '> getRunInfo.py > global name             : ' + Dict_cmsmonRunRegistry['RUN_GLOBALNAME']
if Dict_cmsmonRunRegistry.has_key('RUN_STATUS'):
  print '> getRunInfo.py > status                  : ' + Dict_cmsmonRunRegistry['RUN_STATUS']
if Dict_cmsmonRunRegistry.has_key('RUN_INDBS'):
  print '> getRunInfo.py > in DBS                  : ' + Dict_cmsmonRunRegistry['RUN_INDBS']
if Dict_cmsmonRunRegistry.has_key('subsystems'):
  print '> getRunInfo.py > subsystems              : ' + Dict_cmsmonRunRegistry['subsystems']
if Dict_cmsmonRunRegistry.has_key('RUN_EVENTS'):
  print '> getRunInfo.py > # of triggers           : ' + Dict_cmsmonRunRegistry['RUN_EVENTS']
if Dict_cmsmonRunRegistry.has_key('RUN_START_TIME'):
  print '> getRunInfo.py > start time (local)      : ' + Dict_cmsmonRunRegistry['RUN_START_TIME']
if Dict_cmsmonRunRegistry.has_key('RUN_END_TIME'):
  print '> getRunInfo.py > end time (local)        : ' + Dict_cmsmonRunRegistry['RUN_END_TIME']
if len(str_htmlL1Key) > 0:
  print '> getRunInfo.py > L1 key                  : ' + str_htmlL1Key
if len(str_htmlHLTKey) > 0:
  print '> getRunInfo.py > HLT key                 : ' + str_htmlHLTKey
if Dict_cmsmonRunRegistry.has_key('l1sources'):
  print '> getRunInfo.py > L1 sources              : ' + Dict_cmsmonRunRegistry['l1sources']
if Dict_cmsmonRunRegistry.has_key('RUN_RATE'):
  print '> getRunInfo.py > event rate              : ' + Dict_cmsmonRunRegistry['RUN_RATE'] + ' Hz'
if Dict_cmsmonRunRegistry.has_key('stop_reason'):
  print '> getRunInfo.py > stop reason             : ' + Dict_cmsmonRunRegistry['stop_reason']
if Dict_cmsmonRunRegistry.has_key('RUN_SHIFTER'):
  print '> getRunInfo.py > DQM shifter             : ' + Dict_cmsmonRunRegistry['RUN_SHIFTER']
if Dict_cmsmonRunRegistry.has_key('RUN_COMMENT'):
  print '> getRunInfo.py > DQM shifter\'s comment   : ' + Dict_cmsmonRunRegistry['RUN_COMMENT']
print
# from DBS
print '> getRunInfo.py > * information from DBS *'
print
str_print = '> getRunInfo.py > ' + STR_headDatasets
for int_i in range(int_maxLenDbsDatasets-len(STR_headDatasets)):
  str_print += ' '
str_print += ' '
print str_print + STR_headDatasets
int_len = len(str_print+STR_headDatasets)
str_print = '> '
for int_i in range(int_len-2):
  str_print += '-'
print str_print
for str_dbsDatasets in lstr_dbsDatasets:
  str_print = '                  ' + str_dbsDatasets
  for int_i in range(int_maxLenDbsDatasets-len(str_dbsDatasets)):
    str_print += ' '
  str_print += ' '
  for int_i in range(len(STR_headDatasets)/2-len(Dict_dbsDatasets[str_dbsDatasets])):
    str_print += ' '
  print str_print + Dict_dbsDatasets[str_dbsDatasets]
print  
# from run summary
print '> getRunInfo.py > * information from run summary *'
print
for str_summaryKey in Dict_cmsmonRunSummary.keys():
  print '> getRunInfo.py > ' + str_summaryKey + '\t: ' + Dict_cmsmonRunSummary[str_summaryKey]
print
# from HLT configuration
print '> getRunInfo.py > * information from HLT configuration *'
print
print '> getRunInfo.py > HLT paths included:'
print '> -----------------------------------'
for str_hltPaths in Lstr_hltPaths:
  if str_hltPaths.find('CandHLTTrackerCosmics') >= 0: 
    print '                  ' + str_hltPaths + ' \t<====== FOR SURE!'
  elif str_hltPaths.find('Tracker') >= 0:
    print '                  ' + str_hltPaths + ' \t<====== maybe?'
  else:
    print '                  ' + str_hltPaths
print
# from magnet history
print '> getRunInfo.py > * information from magnet history *'
print
print '> getRunInfo.py > run start time (UTC)    : ' + str(dt_newStart)
print '> getRunInfo.py > run end   time (UTC)    : ' + str(dt_newEnd)
if float_avMagMeasure >= 0.0:
  print '> getRunInfo.py > (average) magnetic field: ' + str(float_avMagMeasure) + ' T'
else:
  print '> getRunInfo.py > cannot determine magnetic field (most probably due to missing time information)' 
print
