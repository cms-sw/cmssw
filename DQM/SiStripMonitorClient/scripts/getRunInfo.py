#!/usr/bin/env python3

#
#

## CMSSW/DQM/SiStripMonitorClient/scripts/getRunInfo.py
#
#  For a given run, this script collects information useful for SiStrip DQM
#  from web sources.
#  Questions and comments to: volker.adler@cern.ch


from __future__ import print_function
import sys
import os
import string
import urllib
import time
import datetime
import getpass

# Constants

# numbers
TD_shiftUTC = datetime.timedelta(hours = 2) # positive for timezones with later time than UTC
INT_offset  = 8
# strings
STR_p5                                  = 'cmsusr0.cern.ch'
STR_wwwWBM                              = 'http://cmswbm/cmsdb/servlet'
STR_SiStrip                             = 'SIST'
STR_wwwDBSData                          = 'dbs_discovery/getData'
LSTR_dbsInstances                       = ['cms_dbs_caf_analysis_01',
                                           'cms_dbs_prod_global'    ]
STR_headDatasets                        = 'datasets'
STR_headFiles                           = 'available data files'
DICT_tagsRunRegistry                    = {}
DICT_tagsRunRegistry['GLOBAL_NAME']     = 'global name                        '
DICT_tagsRunRegistry['STATUS']          = 'status                             '
DICT_tagsRunRegistry['IN_DBS']          = 'in DBS                             '
DICT_tagsRunRegistry['SUBSYSTEMS']      = 'subsystems                         '
DICT_tagsRunRegistry['EVENTS']          = '# of triggers                      '
DICT_tagsRunRegistry['START_TIME']      = 'start time (local)                 '
DICT_tagsRunRegistry['END_TIME']        = 'end time (local)                   '
DICT_tagsRunRegistry['L1KEY']           = 'L1 key                             '
DICT_tagsRunRegistry['HLTKEY']          = 'HLT key                            '
DICT_tagsRunRegistry['L1SOURCES']       = 'L1 sources                         '
DICT_tagsRunRegistry['RUN_RATE']        = 'event rate (Hz)                    '
DICT_tagsRunRegistry['STOP_REASON']     = 'stop reason                        '
DICT_tagsRunRegistry['SHIFTER']         = 'DQM shifter                        '
DICT_tagsRunRegistry['CREATE_USER']     = 'entry created by                   '
DICT_tagsRunRegistry['CREATE_TIME']     = 'entry creation time                '
DICT_tagsRunRegistry['ONLINE_COMMENT']  = 'DQM online shifter\'s comment       '
DICT_tagsRunRegistry['OFFLINE_COMMENT'] = 'DQM offline shifter\'s comment      '
DICT_tagsRunRegistry['OFFLINE_COMMENT'] = 'DQM offline shifter\'s comment      '
DICT_tagsRunRegistry['BFIELD']          = 'magnetic field at run creation time'
DICT_tagsRunRegistry['BFIELD_COMMENT']  = 'comment on magnetic field          '
STR_htlConfig = 'HLT Config ID'
STR_runStart  = 'START_TIME'
STR_runEnd    = 'STOP_TIME'
DICT_keysRunSummary                       = {}
DICT_keysRunSummary[STR_runStart]         = 'start time         '
DICT_keysRunSummary[STR_runEnd]           = 'end time           '
DICT_keysRunSummary['BField']             = 'magnetic field     '
DICT_keysRunSummary['HLT Version']        = 'HLT version        '
DICT_keysRunSummary['L1 Rate']            = 'L1 rate            '
DICT_keysRunSummary['HLT Rate']           = 'HLT rate           '
DICT_keysRunSummary['L1 Triggers']        = 'L1 triggers        '
DICT_keysRunSummary['HLT Triggers']       = 'HLT triggers       '
DICT_keysRunSummary['LHC Fill']           = 'LHC fill           '
DICT_keysRunSummary['LHC Energy']         = 'LHC energy         '
DICT_keysRunSummary['Initial Lumi']       = 'initial luminosity '
DICT_keysRunSummary['Ending Lumi']        = 'ending luminosity  '
DICT_keysRunSummary['Run Lumi']           = 'run luminosity     '
DICT_keysRunSummary['Run Live Lumi']      = 'run live luminosity'
DICT_keysRunSummaryTrigger                = {}
DICT_keysRunSummaryTrigger['L1 Key']      = 'L1 key             '
DICT_keysRunSummaryTrigger['HLT Key']     = 'HLT key            '
DICT_keysRunSummaryTrigger[STR_htlConfig] = 'HLT config ID      '

# Globals

global Str_passwd
global Str_userID
global Str_run
global Dict_runRegistry
global Float_magneticField
global Dict_wbmRunSummary
global Lstr_hltPaths
global DictDict_dbsDatasets
global DictDict_dbsEvents
global Dict_dbsDatasets
global Dict_maxLenDbsDatasets
# initialise
Str_run                = sys.argv[1]
Dict_runRegistry       = {}
Float_magneticField    = -999.0
Dict_wbmRunSummary     = {}
Lstr_hltPaths          = []
DictDict_dbsDatasets   = {}
DictDict_dbsEvents     = {}
Dict_dbsDatasets       = {}
Dict_maxLenDbsDatasets = {}

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
  
## Func_MakeShellWord(str_python)
#
# Adds shell escape charakters to Python strings
def Func_MakeShellWord(str_python):
  """  Func_MakeShellWord(str_python)
  Adds shell escape charakters to Python strings
  """
  return str_python.replace('?','\\?').replace('=','\\=').replace(' ','\\ ').replace('&','\\&').replace(':','\\:')
  
## Func_GetWBMInfo(str_name, str_path)
#
# Logs in on cmsusr0, retrieves WBM information and stores it locally
def Func_GetWBMInfo(str_name, str_path):
  """ Func_GetWBMInfo(str_name, str_path)
  Logs in on cmsusr0, retrieves WBM information and stores it locally
  """
  pid, fd = os.forkpty()
  if pid == 0:
    os.execv('/usr/bin/ssh', ['/usr/bin/ssh', '-l', Str_userID, STR_p5] + ['rm', '-f', '\"'+str_name + '\" && ' + 'wget', '\"'+str_path+'/'+str_name+'\"'])
  else:
    time.sleep(1)
    os.read(fd, 1000)
    time.sleep(1)
    os.write(fd, Str_passwd)
    time.sleep(1)
    c = 0
    s = os.read(fd, 1)
    while s:
      c += 1
      s  = os.read(fd, 1)
      if c >= 2:
        break
  
## Func_CopyWBMInfo(str_name)
#
# Logs in on cmsusr0 and copies file from there
def Func_CopyWBMInfo(str_name):
  """ Func_CopyWBMInfo(str_name)
  Logs in on cmsusr0 and copies file from there
  """
  pid, fd = os.forkpty()
  if pid == 0:
    os.execv('/usr/bin/scp', ['/usr/bin/scp', Str_userID+'@'+STR_p5+':~/'+str_name, '.'])
  else:
    time.sleep(1)
    os.read(fd, 1000)
    time.sleep(1)
    os.write(fd, Str_passwd)
    time.sleep(1)
    c = 0
    s = os.read(fd, 1)
    while s:
      c += 1
      s  = os.read(fd, 1)
      if c >= 163:
        break
  
## Func_FillInfoRunRegistry()
#    
# Retrieves run info from RunRegistry and fills it into containers
def Func_FillInfoRunRegistry():
  """ Func_FillInfoRunRegistry():
  Retrieves run info from RunRegistry and fills it into containers
  """  
  str_runRegistry     = urllib.urlencode({'format':'xml', 'intpl':'xml', 'qtype':'RUN_NUMBER', 'sortname':'RUN_NUMBER'})
  file_runRegistry    = urllib.urlopen("http://pccmsdqm04.cern.ch/runregistry/runregisterdata", str_runRegistry)
  str_runRegistryLong = ''
  for str_runRegistry in file_runRegistry.readlines():
    str_runRegistryLong += str_runRegistry.splitlines()[0]
  bool_foundRun      = False
  str_runRunRegistry = ''
  for int_runIndex in range(1,int(str_runRegistryLong.split('<RUNS')[1].split('>')[0].split('total=\"')[1].split('\"')[0])):
    str_runRunRegistry = Func_GetHtmlTagValue('RUN', str_runRegistryLong, int_runIndex)
    if Func_GetHtmlTagValue('NUMBER', str_runRunRegistry) == Str_run:
      bool_foundRun = True
      break
  if not bool_foundRun:
    print('> getRunInfo.py > run %s not found in run registry' %(Str_run))
    return False
  dict_tagsRunRegistry = Func_GetHtmlTags(str_runRunRegistry)
  for str_tagRunRegistry in dict_tagsRunRegistry.keys():
    if dict_tagsRunRegistry[str_tagRunRegistry] == False:
      Dict_runRegistry[str_tagRunRegistry] = Func_GetHtmlTagValue(str_tagRunRegistry, str_runRunRegistry)
  if Dict_runRegistry['SUBSYSTEMS'].find(STR_SiStrip) < 0:
    print('> getRunInfo.py > SiStrip was not in this run')
    return False
  return True
  
## Func_FillInfoRunSummary()
#    
# Retrieves run info from RunSummary and fills it into containers
def Func_FillInfoRunSummary():
  """ Func_FillInfoRunSummary():
  Retrieves run info from RunSummary and fills it into containers
  """
  str_nameRunSummary = 'RunSummary?RUN=' + Str_run
  Func_GetWBMInfo(str_nameRunSummary, STR_wwwWBM)
  Func_CopyWBMInfo(Func_MakeShellWord(str_nameRunSummary))
  file_wbmRunSummary = file(str_nameRunSummary, 'r')
  bool_table      = False
  int_tableHeader = 0
  int_tableItem   = 0
  int_startItem   = 0
  int_endItem     = 0
  for str_wbmRunSummary in file_wbmRunSummary.readlines():
    if str_wbmRunSummary.find('<TABLE CLASS="params"><THEAD><TR>') >= 0:
      bool_table = True
    if str_wbmRunSummary.find('</TBODY></TABLE>') >= 0:
      bool_table = False
    if bool_table:
      if str_wbmRunSummary.startswith('<TH>'):
        int_tableHeader += 1
        if str_wbmRunSummary.find(STR_runStart) >= 0:
          int_startItem = int_tableHeader
        if str_wbmRunSummary.find(STR_runEnd) >= 0:
          int_endItem = int_tableHeader
      if str_wbmRunSummary.startswith('<TD'):
        int_tableItem += 1
        if int_tableItem == int_startItem:
          Dict_wbmRunSummary[STR_runStart] = str_wbmRunSummary.split('&nbsp;</TD>')[0].split('<TD>')[-1]
        if int_tableItem == int_endItem:
          Dict_wbmRunSummary[STR_runEnd] = str_wbmRunSummary.split('&nbsp;</TD>')[0].split('<TD>')[-1]
      continue
    for str_keyRunSummary in DICT_keysRunSummary.keys():
      if str_wbmRunSummary.find(str_keyRunSummary) >= 0:
        Dict_wbmRunSummary[str_keyRunSummary] = str_wbmRunSummary.split('</TD></TR>')[0].split('>')[-1]
        break
    for str_summaryKeysTrigger in DICT_keysRunSummaryTrigger.keys():
      if str_wbmRunSummary.find(str_summaryKeysTrigger) >= 0:
        Dict_wbmRunSummary[str_summaryKeysTrigger] = str_wbmRunSummary.split('</A></TD></TR>')[0].split('>')[-1]
        if str_summaryKeysTrigger == 'HLT Key':
           Dict_wbmRunSummary[STR_htlConfig] = str_wbmRunSummary.split('HLTConfiguration?KEY=')[1].split('>')[0]
  file_wbmRunSummary.close()
  os.remove(str_nameRunSummary)
  
## Func_FillInfoMagnetHistory()
#    
# Retrieves run info from MagnetHistory and fills it into containers
def Func_FillInfoMagnetHistory(str_timeStart, str_timeEnd):
  """ Func_FillInfoMagnetHistory():
  Retrieves run info from MagnetHistory and fills it into containers
  """
  str_nameMagnetHistory = 'MagnetHistory?TIME_BEGIN=' + str_timeStart + '&TIME_END=' + str_timeEnd
  Func_GetWBMInfo(str_nameMagnetHistory, STR_wwwWBM)
  Func_CopyWBMInfo(Func_MakeShellWord(str_nameMagnetHistory))
  file_wbmMagnetHistory = file(str_nameMagnetHistory, 'r')
  float_avMagMeasure = Float_magneticField
  for str_wbmMagnetHistory in file_wbmMagnetHistory.readlines():
    if str_wbmMagnetHistory.find('BFIELD, Tesla') >= 0:
      float_avMagMeasure = float(str_wbmMagnetHistory.split('</A>')[0].split('>')[-1])
  file_wbmMagnetHistory.close()
  os.remove(str_nameMagnetHistory)
  return float_avMagMeasure
  
## Func_FillInfoHlt()
#    
# Retrieves run info from Hlt and fills it into containers
def Func_FillInfoHlt():
  """ Func_FillInfoHlt():
  Retrieves run info from Hlt and fills it into containers
  """
  str_nameHlt = 'HLTConfiguration?KEY=' + Dict_wbmRunSummary[STR_htlConfig]
  Func_GetWBMInfo(str_nameHlt, STR_wwwWBM)
  Func_CopyWBMInfo(Func_MakeShellWord(str_nameHlt))
  file_wbmHlt     = file(str_nameHlt, 'r')
  bool_foundPaths = False
  bool_foundPath  = False
  for str_wbmHlt in file_wbmHlt.readlines():
    if str_wbmHlt.find('<H3>Paths</H3>') >= 0:
      bool_foundPaths = True
    if bool_foundPaths and str_wbmHlt.find('<HR><H3>') >= 0:
      bool_foundPaths = False
    if bool_foundPaths and str_wbmHlt.startswith('<TR><TD ALIGN=RIGHT>'):
      Lstr_hltPaths.append(str_wbmHlt.split('</TD>')[1].split('<TD>')[-1])
  file_wbmHlt.close()
  os.remove(str_nameHlt)
  return (len(Lstr_hltPaths)>0)
  
## Func_FillInfoDBS(str_dbsInstance)
#
# Retrieves run info from DBS and fills it into containers
def Func_FillInfoDBS(str_dbsInstance):
  """ Func_FillInfoDBS(str_dbsInstance)
  Retrieves run info from DBS and fills it into containers
  """
  str_dbsRuns      = urllib.urlencode({'ajax':'0', '_idx':'0', 'pagerStep':'0', 'userMode':'user', 'release':'Any', 'tier':'Any', 'dbsInst':str_dbsInstance, 'primType':'Any', 'primD':'Any', 'minRun':Str_run, 'maxRun':Str_run})
  file_dbsRuns     = urllib.urlopen("https://cmsweb.cern.ch/dbs_discovery/getRunsFromRange", str_dbsRuns)
  lstr_dbsRuns     = []
  lstr_dbsDatasets = []
  dict_dbsDatasets = {}
  dict_dbsEvents   = {}
  for str_dbsRuns in file_dbsRuns.readlines():
    lstr_dbsRuns.append(str_dbsRuns)
    if str_dbsRuns.find(STR_wwwDBSData) >= 0:
      if str_dbsRuns.split('&amp;proc=')[1].find('&amp;') >= 0:
        lstr_dbsDatasets.append(str_dbsRuns.split('&amp;proc=')[1].split('&amp;')[0])
      else:
        lstr_dbsDatasets.append(str_dbsRuns.split('&amp;proc=')[1])
  int_maxLenDbsDatasets = 0
  for str_dbsDataset in lstr_dbsDatasets:
    str_dbsLFN  = urllib.urlencode({'dbsInst':str_dbsInstance, 'blockName':'*', 'dataset':str_dbsDataset, 'userMode':'user', 'run':Str_run})
    file_dbsLFN = urllib.urlopen("https://cmsweb.cern.ch/dbs_discovery/getLFNlist", str_dbsLFN)
    lstr_dbsLFN = []
    int_events  = 0
    for str_dbsLFN in file_dbsLFN.readlines():
      lstr_dbsLFN.append(str_dbsLFN)
      if str_dbsLFN.find('contians') >= 0 and str_dbsLFN.find('file(s)'): # FIXME: be careful, this typo might be corrected sometimes on the web page...
        dict_dbsDatasets[str_dbsDataset] = str_dbsLFN.split()[1]
      if str_dbsLFN.startswith('/store/data/'):
        int_events += int(Func_GetHtmlTagValue('td' ,lstr_dbsLFN[len(lstr_dbsLFN)-4]))
    dict_dbsEvents[str_dbsDataset] = str(int_events)
    if len(str_dbsDataset) > int_maxLenDbsDatasets:
      int_maxLenDbsDatasets = len(str_dbsDataset)
  DictDict_dbsDatasets[str_dbsInstance]   = dict_dbsDatasets
  DictDict_dbsEvents[str_dbsInstance]     = dict_dbsEvents
  Dict_dbsDatasets[str_dbsInstance]       = lstr_dbsDatasets
  Dict_maxLenDbsDatasets[str_dbsInstance] = int_maxLenDbsDatasets
 
## MAIN PROGRAM

print()
print('> getRunInfo.py > information on run \t*** %s ***' %(Str_run))
print()

# enter online password

Str_userID = getpass.getuser()
Str_passwd = getpass.getpass('> getRunInfo.py > '+Str_userID+'@'+STR_p5+'\'s password: ') + '\n'

# get run RunRegistry entries

bool_runRegistry = Func_FillInfoRunRegistry()

# print run RunRegistry info

if bool_runRegistry:
  print()
  print('> getRunInfo.py > * information from run registry *')
  print()
  for str_htmlTag in DICT_tagsRunRegistry.keys():
    if str_htmlTag in Dict_runRegistry:
      print('> getRunInfo.py > %s: %s' %(DICT_tagsRunRegistry[str_htmlTag],Dict_runRegistry[str_htmlTag]))
  
# get run RunSummary entries

Func_FillInfoRunSummary()

# print run RunSummary info

print()
print('> getRunInfo.py > * information from run summary *')
print()
for str_key in DICT_keysRunSummary.keys():
  if str_key in Dict_wbmRunSummary:
    print('> getRunInfo.py > %s: %s' %(DICT_keysRunSummary[str_key],Dict_wbmRunSummary[str_key]))
for str_key in DICT_keysRunSummaryTrigger.keys():
  if str_key in Dict_wbmRunSummary:
    print('> getRunInfo.py > %s: %s' %(DICT_keysRunSummaryTrigger[str_key],Dict_wbmRunSummary[str_key]))
    
# get run MagnetHistory info

if STR_runStart in Dict_wbmRunSummary and STR_runEnd in Dict_wbmRunSummary: # need run summary start and end time here
  Float_magneticField = Func_FillInfoMagnetHistory(Dict_wbmRunSummary[STR_runStart],Dict_wbmRunSummary[STR_runEnd])
  
# print run MagnetHistory info

if Float_magneticField >= 0.0:
  print()
  print('> getRunInfo.py > * information from magnet history *')
  print()
  print('> getRunInfo.py > (average) magnetic field: %s T' %(str(Float_magneticField)))
  
# get run HLT info

bool_hlt = False   
if STR_htlConfig in Dict_wbmRunSummary: # need HLT config ID from run summary here
  bool_hlt = Func_FillInfoHlt()

# print run HLT info

if bool_hlt:
  print()
  print('> getRunInfo.py > * information from HLT configuration %s *' %(Dict_wbmRunSummary[STR_htlConfig]))
  print()
  print('> getRunInfo.py > HLT paths included:')
  print('> -----------------------------------')
  for str_hltPaths in Lstr_hltPaths:
    if str_hltPaths.find('CandHLTTrackerCosmics') >= 0 or str_hltPaths.find('HLT_TrackerCosmics') >= 0 or str_hltPaths.find('HLTTrackerCosmics') >= 0: 
      print('                  %s \t<====== FOR SURE!' %(str_hltPaths))
    elif str_hltPaths.find('Tracker') >= 0:
      print('                  %s \t<====== maybe?' %(str_hltPaths))
    else:
      print('                  %s' %(str_hltPaths))

# get run DBS entries

for str_dbsInstance in LSTR_dbsInstances:
  Func_FillInfoDBS(str_dbsInstance)

# print run DBS info

print()
print('> getRunInfo.py > * information from DBS *')
for str_dbsInstance in LSTR_dbsInstances:
  print()
  print('> getRunInfo.py > DBS instance: %s' %(str_dbsInstance))
  if str_dbsInstance == LSTR_dbsInstances[0]:
    print('                  (This is the instance used at CAF!)')
  str_print = '> getRunInfo.py > ' + STR_headDatasets
  for int_i in range(Dict_maxLenDbsDatasets[str_dbsInstance]-len(STR_headDatasets)):
    str_print += ' '
  str_print += ' '
  int_length = len(str_print)
  print('%s%s' %(str_print,STR_headFiles))
  str_print = '                  '
  for int_i in range(int_length-16+len(STR_headFiles)/2+INT_offset+8):
    str_print += '-'
  print(str_print)
  for str_dbsDataset in Dict_dbsDatasets[str_dbsInstance]:
    str_print = '                  ' + str_dbsDataset
    for int_i in range(Dict_maxLenDbsDatasets[str_dbsInstance]-len(str_dbsDataset)):
      str_print += ' '
    str_print += ' '
    for int_i in range(len(STR_headFiles)/2-len(DictDict_dbsDatasets[str_dbsInstance][str_dbsDataset])):
      str_print += ' '
    str_print += DictDict_dbsDatasets[str_dbsInstance][str_dbsDataset] + ' ('
    for int_i in range(INT_offset-len(DictDict_dbsEvents[str_dbsInstance][str_dbsDataset])):
      str_print += ' '
    print('%s%s events)' %(str_print,DictDict_dbsEvents[str_dbsInstance][str_dbsDataset]))

print()  
