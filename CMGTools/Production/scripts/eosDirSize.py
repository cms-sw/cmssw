#!/bin/env python

import sys
import CMGTools.Production.eostools as eostools

print eostools.eosDirSize( sys.argv[1] )
