#!/usr/bin/env python3

# For documentation of the RR XML-RPC handler, look into https://twiki.cern.ch/twiki//bin/view/CMS/DqmRrApi

from __future__ import print_function
import sys
import xmlrpclib


def displayHelp():
  print("""
  getRunRegistry.py

  CMSSW package DQM/TrackerCommon

  Usage:
  $ getRunRegistry.py [ARGUMENTOPTION1] [ARGUMENT1] ... [OPTION2] ...

  Valid argument options are:
    -s
      API address of RunRegistry server
      default: 'http://pccmsdqm04.cern.ch/runregistry/xmlrpc'
    -T
      table identifier
      available: 'RUN', 'RUNLUMISECTION'
      default: 'RUN'
    -w
      work space identifier
      available: 'RPC', 'HLT', 'L1T', 'TRACKER', 'CSC', 'GLOBAL', 'ECAL'
      default: 'GLOBAL'
    -t
      output format type
      available:
        - table 'RUN'           : 'chart_runs_cum_evs_vs_bfield', 'chart_runs_cum_l1_124_vs_bfield', 'chart_stacked_component_status', 'csv_datasets', 'csv_run_numbers', 'csv_runs', 'tsv_datasets', 'tsv_runs', 'xml_all', 'xml_datasets'
        - table 'RUNLUMISECTION': 'json', 'xml'
      default: 'xml_all' (for table 'RUN')
    -f
      output file
      default: 'runRegistry.xml'
    -l
      lower bound of run numbers to consider
      default: '0'
    -u
      upper bound of run numbers to consider
      default: '1073741824'

  Valid options are:
    -h
      display this help and exit
  """)


# Option handling (very simple, no validity checks)
sOptions = {
  '-s': 'http://pccmsdqm04.cern.ch/runregistry/xmlrpc' # RunRegistry API proxy server
, '-T': 'RUN'                                          # table
, '-w': 'GLOBAL'                                       # workspace
, '-t': 'xml_all'                                      # output format type
, '-f': 'runRegistry.xml'                              # output file
, '-l': '0'                                            # lower bound of run numbers to consider
, '-u': '1073741824'                                   # upper bound of run numbers to consider
}
bOptions = {
  '-h': False # help option
}
iArgument  = 0
for token in sys.argv[ 1:-1 ]:
  iArgument = iArgument + 1
  if token in sOptions.keys():
    if not sys.argv[ iArgument + 1 ] in sOptions.keys() and not sys.argv[ iArgument + 1 ] in bOptions.keys():
      del sOptions[ token ]
      sOptions[ token ] = sys.argv[ iArgument + 1 ]
for token in sys.argv[ 1: ]:
  if token in bOptions.keys():
    del bOptions[ token ]
    bOptions[ token ] = True
if bOptions[ '-h' ]:
  displayHelp()
  sys.exit( 0 )

# Data extraction and local storage
# initialise API access to defined RunRegistry proxy
server = xmlrpclib.ServerProxy( sOptions[ '-s' ] )
# get data according to defined table, workspace and output format type
runs = '{runNumber} >= ' + sOptions[ '-l' ] + 'and {runNumber} <= ' + sOptions[ '-u' ]
data = server.DataExporter.export( sOptions[ '-T' ], sOptions[ '-w' ], sOptions[ '-t' ], runs )
# write data to file
file = open( sOptions[ '-f' ], 'w' )
file.write( data )
file.close()
