#! /bin/env python
import sys

import CMGTools.Production.eostools as eostools

# print sys.argv[1]
eostools.xrdcp( sys.argv[1], sys.argv[2] )

# eostools.listFiles('/eos/cms/store/cmst3/user/cbern/Tests/', rec=True)

# sys.exit(0)
