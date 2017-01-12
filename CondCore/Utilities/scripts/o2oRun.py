#!/usr/bin/env python
'''
'''

__author__ = 'Giacomo Govi'

import CondCore.Utilities.o2o as o2olib
import sys
import optparse
import argparse
    
def main( argv ):

    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=str, help="wrapper for O2O jobs")
    parser.add_argument("-n","--name", type=str, help="the O2O job name" )
    parser.add_argument("-d","--dev", action="store_true", help="bookkeeping in dev database")
    parser.add_argument("-p","--private", action="store_true", help="bookkeeping in private database")
    parser.add_argument("-a","--auth", type=str,  help="path of the authentication file")
    args = parser.parse_args()  

    if not args.name:
        parser.error("Job name not given.")

    command = args.executable

    db_service = None
    if not args.private:
        if args.dev:
            db_service = o2olib.dev_db_service
        else:
            db_service = o2olib.prod_db_service
    runMgr = o2olib.O2ORunMgr()
    ret = -1
    if runMgr.connect( db_service, args.auth ):
        ret = runMgr.executeJob( args.name, command )
    return ret

if __name__ == '__main__':
    sys.exit(main(sys.argv))
