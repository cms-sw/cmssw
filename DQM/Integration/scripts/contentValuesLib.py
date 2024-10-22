from __future__ import print_function
import sys, os, optparse, re, json
import ROOT, xmlrpclib
 
SERVER_URL = "http://pccmsdqm04.cern.ch/runregistry/xmlrpc"

ONLINE_DATASET = '/Global/Online/ALL'

# FOLDER UNDER EvInfo        NAME    SUMMARY VALUE IN EvInfo
FOLDERS     = { 
  'reportSummaryContents': ( 'DQM',  'reportSummary' ),
  'CertificationContents': ( 'CERT', 'CertificationSummary' ),
  'DAQContents':           ( 'DAQ',  'DAQSummary' ),
  'DCSContents':           ( 'DCS',  'DCSSummary' )
}

SUBSYSTEMS  = {
  'CSC' :        'CSC',
  'DT' :         'DT',
  'ES' :         'ES',
  'EcalBarrel' : 'ECAL',
  'EcalEndcap' : 'ECAL',
  'Hcal' :       'HCAL',
  'L1T' :        'L1T',
  'L1TEMU' :     'L1T',
  'Pixel' :      'PIX',
  'RPC' :        'RPC',
  'SiStrip' :    'STRIP'
}

def getDatasetName(file_name):
  """ Method to get dataset name from the file name"""
  d = None
  try:
    d = re.search("(__[a-zA-Z0-9-_]+)+", file_name).group(0)
    d = re.sub("__", "/", d)
  except:
    d = None
  return d

def getSummaryValues(file_name, translate, filters = None):
  """ Method to extract keys from root file and return dict """
  ROOT.gROOT.Reset()

  run_number = None
  result = {}

  f = ROOT.TFile(file_name, 'READ')

  root = f.GetDirectory("DQMData")
  if root == None: return (run_number, result)
  
  run = None
  for key in root.GetListOfKeys():
    if re.match("^Run [0-9]+$", key.ReadObj().GetName()) and key.IsFolder():
      run_number = int(re.sub("^Run ", "", key.ReadObj().GetName()))
      run = key.ReadObj()
      break

  if run == None: return (run_number, result)

  for sub in run.GetListOfKeys():

    sub_name = sub.ReadObj().GetName()
    if sub_name not in SUBSYSTEMS: continue

    sub_key = sub_name
    if translate:
      sub_key = SUBSYSTEMS[sub_name]

    if filters != None:
      if not re.match(filters[0], sub_key):
        continue
    
    if sub_key not in result:
      result[sub_key] = {}

    evInfo = sub.ReadObj().GetDirectory("Run summary/EventInfo")
    if evInfo == None: continue

    for folder_name in FOLDERS.keys():

      folder = evInfo.GetDirectory(folder_name)
      if folder == None: continue
      
      folder_id = folder_name
      if translate:
        folder_id = FOLDERS[folder_name][0]
      
      if filters != None:
        if not re.match(filters[1], folder_id):
          continue
    
      if folder_id not in result[sub_key]:
        result[sub_key][folder_id] = {}

      value_filter = None
      if filters != None:
        value_filter = filters[2]

      writeValues(folder, result[sub_key][folder_id], None, value_filter)
      writeValues(evInfo, result[sub_key][folder_id], {FOLDERS[folder_name][1]: 'Summary'}, value_filter)

  f.Close()

  return (run_number, result)

def writeValues(folder, map, keymap = None, filter = None):
  """ Write values (possibly only for the keys in the keymap and filtered) from folder to map """
  for value in folder.GetListOfKeys():
    full_name = value.ReadObj().GetName()
    if not value.IsFolder() and re.match("^<.+>f=-{,1}[0-9\.]+</.+>$", full_name):
      value_name = re.sub("<(?P<n>[^>]+)>.+", "\g<n>", full_name)
      value_numb = float(re.sub("<.+>f=(?P<n>-{,1}[0-9\.]+)</.+>", "\g<n>", full_name))
      if keymap == None or value_name in keymap:
        if not keymap == None:
          if not keymap[value_name] == None:
            value_name = keymap[value_name]
        if filter == None or re.match(filter, value_name):
          if value_name not in map:
            map[value_name] = value_numb

def checkFilter(raw_filter):
  """ Check if filter is OK """
  if raw_filter != None:
    try:
      filter = eval(raw_filter)
      if not isinstance("", type(filter[0])) or not isinstance("", type(filter[1])) or not isinstance("", type(filter[2])):
        raise TypeError('')
    except:
      print("Bad filter value ", raw_filter, ".\nFilter should be written in python tupple with 3 elements, i.e. \"('subsystem','folder','value')\". elements are in regexp format.")
      sys.exit(2)
  else:
    filter = ('.*', '.*', '.*')
  return filter
