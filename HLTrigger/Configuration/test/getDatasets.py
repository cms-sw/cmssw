#!/usr/bin/env python3
"""getDatasets.py: create Datasets-cff file of an HLT configuration from the ConfDB database
"""
import argparse
import subprocess
import os
import re

import FWCore.ParameterSet.Config as cms
import HLTrigger.Configuration.Tools.pipe as pipe
import HLTrigger.Configuration.Tools.options as options

def getHLTProcess(config):
  '''return cms.Process containing Streams and Datasets of the HLT configuration
  '''
  # cmd-line args to select HLT configuration
  if config.menu.run:
    configline = f'--runNumber {config.menu.run}'
  else:
    configline = f'--{config.menu.database} --{config.menu.version} --configName {config.menu.name}'

  # cmd to download HLT configuration
  cmdline = f'hltConfigFromDB {configline} --noedsources --noes --noservices --nopsets --nooutput --nopaths'
  if config.proxy:
    cmdline += f' --dbproxy --dbproxyhost {config.proxy_host} --dbproxyport {config.proxy_port}'

  # load HLT configuration
  try:
    foo = {'process': None}
    exec(pipe.pipe(cmdline).decode(), foo)
    process = foo['process']
  except:
    raise Exception(f'query did not return a valid python file:\n query="{cmdline}"')

  if not isinstance(process, cms.Process):
    raise Exception(f'query did not return a valid HLT menu:\n query="{cmdline}"')

  return process

###
### main
###
if __name__ == '__main__':

  # defaults of cmd-line arguments
  defaults = options.HLTProcessOptions()

  parser = argparse.ArgumentParser(
    prog = './'+os.path.basename(__file__),
    formatter_class = argparse.RawDescriptionHelpFormatter,
    description = __doc__)

  # required argument
  parser.add_argument('menu',
                      action  = 'store',
                      type    = options.ConnectionHLTMenu,
                      metavar = 'MENU',
                      help    = 'HLT menu to dump from the database. Supported formats are:\n  - /path/to/configuration[/Vn]\n  - [[{v1|v2|v3}/]{run3|run2|online|adg}:]/path/to/configuration[/Vn]\n  - run:runnumber\nThe possible converters are "v1", "v2, and "v3" (default).\nThe possible databases are "run3" (default, used for offline development), "run2" (used for accessing run2 offline development menus), "online" (used to extract online menus within Point 5) and "adg" (used to extract the online menus outside Point 5).\nIf no menu version is specified, the latest one is automatically used.\nIf "run:" is used instead, the HLT menu used for the given run number is looked up and used.\nNote other converters and databases exist as options but they are only for expert/special use.' )

  # options
  parser.add_argument('--dbproxy',
                      dest    = 'proxy',
                      action  = 'store_true',
                      default = defaults.proxy,
                      help    = 'Use a socks proxy to connect outside CERN network (default: False)' )
  parser.add_argument('--dbproxyport',
                      dest    = 'proxy_port',
                      action  = 'store',
                      metavar = 'PROXYPORT',
                      default = defaults.proxy_port,
                      help    = 'Port of the socks proxy (default: 8080)' )
  parser.add_argument('--dbproxyhost',
                      dest    = 'proxy_host',
                      action  = 'store',
                      metavar = 'PROXYHOST',
                      default = defaults.proxy_host,
                      help    = 'Host of the socks proxy (default: "localhost")' )

  # parse command line arguments and options
  config = parser.parse_args()

  process = getHLTProcess(config)

  print('''# %s

import FWCore.ParameterSet.Config as cms
''' % config.menu.name)

  for stream in sorted(process.streams.__dict__):
    if re.match(r'^(Physics|Parking)', stream):
      print('''
# stream %s
''' % stream)
      ds = sorted(process.streams.__dict__[stream])
      for dataset in ds:
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
        print(dump)
