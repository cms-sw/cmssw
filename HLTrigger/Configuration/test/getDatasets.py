#! /usr/bin/env python

import sys
import subprocess
import imp
import re
import FWCore.ParameterSet.Config as cms

def extractDatasets(version, database, config):
  # dump the streams and Datasets from the HLT configuration
  proc = subprocess.Popen(
    "hltConfigFromDB --%s --%s --configName %s --nopsets --noedsources --noes --noservices --nooutput --nopaths" % (version, database, config),
    shell  = True,
    stdin  = None,
    stdout = subprocess.PIPE,
    stderr = None,
  )
  (out, err) = proc.communicate()

  # load the streams and Datasets
  hlt = imp.new_module('hlt')
  exec(out, globals(), hlt.__dict__)

  return hlt.process


def dumpDataset(process, stream, dataset):
  if dataset in process.datasets.__dict__:
    name = 'stream%s_dataset%s_selector' % (stream, dataset)
    dump = '''from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as %s
%s.hltResults = cms.InputTag('TriggerResults', '', 'HLT')
%s.l1tResults = cms.InputTag('')
%s.throw      = cms.bool(False)
%s.triggerConditions = %s

''' % (name, name, name, name, name, process.datasets.__dict__[dataset])
  else:
    dump = '''# dataset %s not found

''' % (dataset, )
  return dump


# split a "[version/]db:name" configuration into a (version, db, name) tuple 
def splitConfigName(configName):
  from HLTrigger.Configuration.Tools.options import ConnectionHLTMenu
  menu = ConnectionHLTMenu(configName)
  return (menu.version, menu.database, menu.name)


# get the configuration to parse and the file where to output the stream definitions from the command line
config = sys.argv[1]

# dump the expanded event content configurations to a python configuration fragment
config  = splitConfigName(config)
process = extractDatasets(* config)

sys.stdout.write('''# %s

import FWCore.ParameterSet.Config as cms

''' % config[2] )

for stream in sorted(process.streams.__dict__):
  if re.match(r'^Physics|Parking', stream):
    sys.stdout.write('''
# stream %s

''' % stream)
    ds = sorted(process.streams.__dict__[stream])
    for dataset in ds:
      sys.stdout.write(dumpDataset(process, stream, dataset))
