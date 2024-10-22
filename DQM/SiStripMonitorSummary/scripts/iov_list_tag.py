#!/usr/bin/env python3
from __future__ import print_function
from optparse import OptionParser, Option, OptionValueError
import DLFCN
import sys
sys.setdlopenflags(DLFCN.RTLD_GLOBAL+DLFCN.RTLD_LAZY)

import pluginCondDBPyInterface as condDB
a = condDB.FWIncantation()

def list_tag( conn_str, tag, auth_path ):
    rdbms = condDB.RDBMS( auth_path )
    db = rdbms.getReadOnlyDB( conn_str )
    db.startReadOnlyTransaction()
    iov = db.iov( tag )
    for elem in iov.elements:
        print(elem.since())
        db.commitTransaction()
        
if __name__ == "__main__":
    parser=OptionParser()
    parser.add_option("-c","--connection",action="store",dest="conn_str",help="connection string of the target account")
    parser.add_option("-t","--tag",action="store",dest="tag",help="tag to print")
    parser.add_option("-P","--auth_path",action="store",dest="auth_path",help="authentication path")
    (options, args) = parser.parse_args()
    list_tag( parser.values.conn_str,parser.values.tag, parser.values.auth_path  )

            
