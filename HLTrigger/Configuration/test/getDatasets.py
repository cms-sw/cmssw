#! /usr/bin/env python

import sys
import subprocess
import imp
import FWCore.ParameterSet.Config as cms

def extractDatasets(database, config):
  # dump the Streams and Datasets from the HLT configuration
  proc = subprocess.Popen(
    "hltConfigFromDB --%s --configName %s --nopsets --noedsources --noes --noservices --nooutput --nopaths" % (database, config),
    shell  = True,
    stdin  = None,
    stdout = subprocess.PIPE,
    stderr = None,
  )
  (out, err) = proc.communicate()

  # load the Streams and Datasets
  hlt = imp.new_module('hlt')
  exec out in globals(), hlt.__dict__

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


# split a "db:name" configuration into a (db, name) tuple 
def splitConfigName(configName):
  # extract the database and configuration name
  if ':' in configName:
    (menuConfigDB, menuConfigName) = configName.split(':')
    if menuConfigDB not in ('hltdev', 'orcoff'):
      print 'Unknown ConfDB database "%s", valid values are "hltdev" (default) and "orcoff")' % menuConfigDB
      sys.exit(1)
  else:
    (menuConfigDB, menuConfigName) = ('hltdev', configName)

  return (menuConfigDB, menuConfigName)


# get the configuration to parse and the file where to output the stream definitions from the command line
config = sys.argv[1]
target = sys.argv[2]

# dump the expanded event content configurations to a python configuration fragment
process = extractDatasets( * splitConfigName(config) )

dump = open(target, 'w')
dump.write('''# %s

import FWCore.ParameterSet.Config as cms

''' % "getDatasets.py" )

list_of_streams = ['A','AForPP','AForHI','PhysicsEGammaCommissioning','PhysicsHadronsTaus','PhysicsMuons']
for stream in list_of_streams:
  if stream  in process.streams.__dict__:
    dump.write('''
# dump of the Stream %s Datasets defined in the HLT table as Stream A Datasets

''' % stream)
    ds = process.streams.__dict__[stream]
    ds.sort()
    for dataset in ds:
      dump.write( dumpDataset(process, 'A', dataset) )

dump.close()
