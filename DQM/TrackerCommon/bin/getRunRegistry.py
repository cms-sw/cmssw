#!/usr/bin/env python

# For documentation of the RR XML-RPC handler, look into https://twiki.cern.ch/twiki//bin/view/CMS/DqmRrApi

import sys
import xmlrpclib

# Option handling (very simple, no validity checks)
dArguments = { '-s': 'http://pccmsdqm04.cern.ch/runregistry/xmlrpc', # RunRegistry API proxy server
               '-w': 'GLOBAL'                                      , # workspace
               '-t': 'xml_all'                                     , # output format type
               '-f': 'runRegistry.xml'                             } # output file
iArgument  = 0
for argument in sys.argv[ 1:-1 ]:
  iArgument = iArgument + 1
  if argument in dArguments.keys():
    if not sys.argv[ iArgument + 1 ] in dArguments.keys():
      del dArguments[ argument ]
      dArguments[ argument ] = sys.argv[ iArgument + 1 ]

# Data extraction and local storage
server = xmlrpclib.ServerProxy( dArguments[ '-s' ] )                            # initialise API access to defined RunRegistry proxy
data = server.DataExporter.export( 'RUN', dArguments[ '-w' ], dArguments[ '-t' ], {} ) # get data according to defined workspave and output format type
file = open( dArguments[ '-f' ], 'w' )                                          # open defined output file in (over-)write mode
file.write( data )
file.close()
