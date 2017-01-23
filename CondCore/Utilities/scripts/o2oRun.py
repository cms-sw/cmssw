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
    parser.add_argument("--db", type=str, help="the target database: pro ( for prod ) or dev ( for prep ). default=pro")
    parser.add_argument("-a","--auth", type=str,  help="path of the authentication file")
    parser.add_argument("-v","--verbose", action="store_true", help="job output mirrored to screen (default=logfile only)")
    args = parser.parse_args()  

    if not args.name:
        parser.error("Job name not given.")

    db_service = o2olib.prod_db_service
    if args.db is not None:
        if args.db == 'dev' or args.db == 'oradev' :
            db_service = o2olib.dev_db_service
        elif args.db != 'orapro' and args.db != 'onlineorapro' and args.db != 'pro':
            raise Exception("Database '%s' is not known." %self.args.db )
    runMgr = o2olib.O2ORunMgr()
    ret = -1
    if runMgr.connect( db_service, args ):
        ret = runMgr.executeJob( args )
    return ret

if __name__ == '__main__':
    sys.exit(main(sys.argv))
