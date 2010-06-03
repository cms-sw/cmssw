#!/usr/bin/env python

# For documentation of the RR XML-RPC handler, look into https://twiki.cern.ch/twiki//bin/view/CMS/DqmRrApi

import sys
import xmlrpclib

# Option handling (very simple, no validity checks)
dArguments = { '-s': 'http://pccmsdqm04.cern.ch/runregistry/xmlrpc', # RunRegistry API proxy server
               '-T': 'RUN'                                         , # table
               '-w': 'GLOBAL'                                      , # workspace
               '-t': 'xml_all'                                     , # output format type
               '-f': 'runRegistry.xml'                             , # output file
               '-l': '0'                                           , # lower bound of run numbers to consider
               '-u': '1073741824'                                  } # upper bound of run numbers to consider
iArgument  = 0
for argument in sys.argv[ 1:-1 ]:
  iArgument = iArgument + 1
  if argument in dArguments.keys():
    if not sys.argv[ iArgument + 1 ] in dArguments.keys():
      del dArguments[ argument ]
      dArguments[ argument ] = sys.argv[ iArgument + 1 ]

# Data extraction and local storage
# initialise API access to defined RunRegistry proxy
server = xmlrpclib.ServerProxy( dArguments[ '-s' ] )
# get data according to defined table, workspace and output format type
runs = '{runNumber} >= ' + dArguments[ '-l' ] + 'and {runNumber} <= ' + dArguments[ '-u' ]
data = server.DataExporter.export( dArguments[ '-T' ], dArguments[ '-w' ], dArguments[ '-t' ], runs )
# write data to file
file = open( dArguments[ '-f' ], 'w' )
file.write( data )
file.close()
