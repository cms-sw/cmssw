#!/usr/bin/env python

#
# $Id:$
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
INT_offset  = 8
# strings
STR_SiStrip             = 'SIST'
STR_wwwDBSData          = 'https://cmsweb.cern.ch/dbs_discovery/getData'
STR_headDatasets        = 'datasets'
STR_headFiles           = 'available data files'
LSTR_summaryKeys        = ['BField', 'HLT Version', 'L1 Rate', 'HLT Rate', 'L1 Triggers', 'HLT Triggers', 'LHC Fill', 'LHC Energy', 'Initial Lumi', 'Ending Lumi', 'Run Lumi', 'Run Live Lumi']
LSTR_summaryKeysTrigger = ['L1 Key', 'HLT Key']   

# Globals

global Str_run
global Dict_cmsmonRunRegistry
global Dict_cmsmonRunSummary
global Dict_dbsDatasets
global Dict_dbsEvents
global Lstr_hltPaths
# initialise
Str_run                = sys.argv[1]
Dict_cmsmonRunRegistry = {}
Dict_cmsmonRunSummary  = {}
Dict_dbsDatasets       = {}
Dict_dbsEvents         = {}
Lstr_hltPaths          = []

## FUNCTIONS

## Func_GetHtmlTags(str_text)
#
# Gets HTML tags from a string
def Func_GetHtmlTags(str_text):
  """  Func_GetHtmlTags(str_text):
  Gets HTML tags from a string
  """
  dict_tags  = {}
  # first look for tags w/ values
  lstr_split = str_text.split('</')
  for str_split in lstr_split[1:]:
    str_key            = str_split.split('>')[0]
    dict_tags[str_key] = str_key in dict_tags
  # second look for tags w/o values
  lstr_split = str_text.split('/>')
  for str_split in lstr_split[:-1]:
    str_key            = str_split.split('<')[-1].split()[0]
    dict_tags[str_key] = str_key in dict_tags
  return dict_tags
 
## Func_GetHtmlTagValue(str_tag, str_text)
#
# Gets the value of the n-th oocurence a given HTML tag from a string
def Func_GetHtmlTagValue(str_tag, str_text, int_index = 1):
  """  Func_GetHtmlTagValue(str_tag, str_text):
   Gets the value of the n-th oocurence a given HTML tag from a string
  """
  if int_index > str_text.count('<'+str_tag):
    return ''
  str_1 = str_text.split('<'+str_tag)[int_index]
  if str_1[0] != '>':
    if str_1.split('>')[0][-1] == '/':
      return ''
  return str_1.split('>',1)[1].split('</'+str_tag+'>')[0]

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
# Gets the (last) attribute of a given HTML tag value from a string
def Func_GetHtmlTagValueAttr(str_value, str_text):
  """  Func_GetHtmlTagValueAttr(str_value, str_text):
  Gets the (last) attributes of a given HTML tag value from a string
  """
  return str_text.split('\">'+str_value+'<')[0].split('=\"')[-1]
  
## Func_FillInfoRunRegistry()
#    
# Retrieves run info from RunRegistry and fills it into containers
def Func_FillInfoRunRegistry():
  """ Func_FillInfoRunRegistry():
  Retrieves run info from RunRegistry and fills it into containers
  """  
  str_cmsmonRunRegistry     = urllib.urlencode({'format':'xml', 'intpl':'xml', 'qtype':'RUN_NUMBER', 'sortname':'RUN_NUMBER'})
  file_cmsmonRunRegistry    = urllib.urlopen("http://pccmsdqm04.cern.ch/runregistry/runregisterdata", str_cmsmonRunRegistry)
  str_cmsmonRunRegistryLong = ''
  for str_cmsmonRunRegistry in file_cmsmonRunRegistry.readlines():
    str_cmsmonRunRegistryLong += str_cmsmonRunRegistry.splitlines()[0]
  bool_foundRun = False
  str_cmsmonRun = ''
  for int_runIndex in range(1,int(str_cmsmonRunRegistryLong.split('<RUNS')[1].split('>')[0].split('total=\"')[1].split('\"')[0])):
    str_cmsmonRun = Func_GetHtmlTagValue('RUN', str_cmsmonRunRegistryLong, int_runIndex)
    if Func_GetHtmlTagValue('NUMBER', str_cmsmonRun) == Str_run:
      bool_foundRun = True
      break
  if not bool_foundRun:
    print '> getRunInfo.py > run ' + Str_run + ' not found in run registry'
    return False
  dict_cmsmonHtmlTags = Func_GetHtmlTags(str_cmsmonRun)
  for str_cmsmonHtmlTag in dict_cmsmonHtmlTags.keys():
    if dict_cmsmonHtmlTags[str_cmsmonHtmlTag] == False:
      Dict_cmsmonRunRegistry[str_cmsmonHtmlTag] = Func_GetHtmlTagValue(str_cmsmonHtmlTag, str_cmsmonRun)
  if Dict_cmsmonRunRegistry['SUBSYSTEMS'].find(STR_SiStrip) < 0:
    print '> getRunInfo.py > SiStrip was not in this run'
    return False
  return True
 
## MAIN PROGRAM

print
print '> getRunInfo.py > information on run \t*** ' + Str_run + ' ***'
print

# Get run information from the web

# get run RunRegistry entries
bool_runRegistry = Func_FillInfoRunRegistry()

# get run DBS entries
str_dbsRuns      = urllib.urlencode({'ajax':'0', '_idx':'0', 'pagerStep':'0', 'userMode':'user', 'release':'Any', 'tier':'Any', 'dbsInst':'cms_dbs_caf_analysis_01', 'primType':'Any', 'primD':'Any', 'minRun':Str_run, 'maxRun':Str_run})
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
  lstr_dbsLFN = []
  int_events  = 0
  for str_dbsLFN in file_dbsLFN.readlines():
    lstr_dbsLFN.append(str_dbsLFN)
    if str_dbsLFN.find('contians') >= 0 and str_dbsLFN.find('file(s)'): # FIXME: be careful, this typo might be corrected sometimes on the web page...
      Dict_dbsDatasets[str_dbsDatasets] = str_dbsLFN.split()[1]
    if str_dbsLFN.startswith('/store/data/'):
      int_events += int(Func_GetHtmlTagValue('td' ,lstr_dbsLFN[len(lstr_dbsLFN)-4]))
  Dict_dbsEvents[str_dbsDatasets] = str(int_events)
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

# Determine further information

# get magnetic field
float_avMagMeasure = -999.0
dt_newStart        = datetime.datetime(2000,1,1,0,0,0)
dt_newEnd          = datetime.datetime(2000,1,1,0,0,0)
if ( Dict_cmsmonRunRegistry.has_key('START_TIME') and Dict_cmsmonRunRegistry.has_key('END_TIME') ):
  lstr_dateStart = Dict_cmsmonRunRegistry['START_TIME'].split(' ')[0].split('.')
  lstr_timeStart = Dict_cmsmonRunRegistry['START_TIME'].split(' ')[1].split(':')
  lstr_dateEnd   = Dict_cmsmonRunRegistry['END_TIME'].split(' ')[0].split('.')
  lstr_timeEnd   = Dict_cmsmonRunRegistry['END_TIME'].split(' ')[1].split(':')
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
if bool_runRegistry:
  print
  print '> getRunInfo.py > * information from run registry *'
  print
  if 'GLOBAL_NAME' in Dict_cmsmonRunRegistry:
    print '> getRunInfo.py > global name                  : ' + Dict_cmsmonRunRegistry['GLOBAL_NAME']
  if 'STATUS' in Dict_cmsmonRunRegistry:
    print '> getRunInfo.py > status                       : ' + Dict_cmsmonRunRegistry['STATUS']
  if 'IN_DBS' in Dict_cmsmonRunRegistry:
    print '> getRunInfo.py > in DBS                       : ' + Dict_cmsmonRunRegistry['IN_DBS']
  if 'SUBSYSTEMS' in Dict_cmsmonRunRegistry:
    print '> getRunInfo.py > subsystems                   : ' + Dict_cmsmonRunRegistry['SUBSYSTEMS']
  if 'EVENTS' in Dict_cmsmonRunRegistry:
    print '> getRunInfo.py > # of triggers                : ' + Dict_cmsmonRunRegistry['EVENTS']
  if 'START_TIME' in Dict_cmsmonRunRegistry:
    print '> getRunInfo.py > start time (local)           : ' + Dict_cmsmonRunRegistry['START_TIME']
  if 'END_TIME' in Dict_cmsmonRunRegistry:
    print '> getRunInfo.py > end time (local)             : ' + Dict_cmsmonRunRegistry['END_TIME']
  if 'L1KEY' in Dict_cmsmonRunRegistry:
    print '> getRunInfo.py > L1 key                       : ' + Dict_cmsmonRunRegistry['L1KEY']
  if 'HLTKEY' in Dict_cmsmonRunRegistry:
    print '> getRunInfo.py > HLT key                      : ' + Dict_cmsmonRunRegistry['HLTKEY']
  if 'L1SOURCES' in Dict_cmsmonRunRegistry:
    print '> getRunInfo.py > L1 sources                   : ' + Dict_cmsmonRunRegistry['L1SOURCES']
#   if 'RUN_RATE' in Dict_cmsmonRunRegistry:
#     print '> getRunInfo.py > event rate                   : ' + Dict_cmsmonRunRegistry['RUN_RATE'] + ' Hz'
  if 'STOP_REASON' in Dict_cmsmonRunRegistry:
    print '> getRunInfo.py > stop reason                  : ' + Dict_cmsmonRunRegistry['STOP_REASON']
  if 'SHIFTER' in Dict_cmsmonRunRegistry:
    print '> getRunInfo.py > DQM shifter                  : ' + Dict_cmsmonRunRegistry['SHIFTER']
  if 'CREATE_USER' in Dict_cmsmonRunRegistry:
    print '> getRunInfo.py > entry created by             : ' + Dict_cmsmonRunRegistry['CREATE_USER']
  if 'CREATE_TIME' in Dict_cmsmonRunRegistry:
    print '> getRunInfo.py > entry creation time          : ' + Dict_cmsmonRunRegistry['CREATE_TIME']
  if 'ONLINE_COMMENT' in Dict_cmsmonRunRegistry:
    print '> getRunInfo.py > DQM online shifter\'s comment : ' + Dict_cmsmonRunRegistry['ONLINE_COMMENT']
  if 'OFFLINE_COMMENT' in Dict_cmsmonRunRegistry:
    print '> getRunInfo.py > DQM offline shifter\'s comment: ' + Dict_cmsmonRunRegistry['OFFLINE_COMMENT']
print

# from DBS
print '> getRunInfo.py > * information from DBS *'
print
str_print = '> getRunInfo.py > ' + STR_headDatasets
for int_i in range(int_maxLenDbsDatasets-len(STR_headDatasets)):
  str_print += ' '
str_print += ' '
int_length = len(str_print)
print str_print + STR_headFiles
str_print = '> '
for int_i in range(int_length+len(STR_headFiles)/2+INT_offset+8):
  str_print += '-'
print str_print
for str_dbsDatasets in lstr_dbsDatasets:
  str_print = '                  ' + str_dbsDatasets
  for int_i in range(int_maxLenDbsDatasets-len(str_dbsDatasets)):
    str_print += ' '
  str_print += ' '
  for int_i in range(len(STR_headFiles)/2-len(Dict_dbsDatasets[str_dbsDatasets])):
    str_print += ' '
  str_print += Dict_dbsDatasets[str_dbsDatasets] + ' ('
  for int_i in range(INT_offset-len(Dict_dbsEvents[str_dbsDatasets])):
    str_print += ' '
  print str_print + Dict_dbsEvents[str_dbsDatasets] + ' events)'
print
  
# from RunSummary
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
  if str_hltPaths.find('CandHLTTrackerCosmics') >= 0 or str_hltPaths.find('HLT_TrackerCosmics') >= 0: 
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
