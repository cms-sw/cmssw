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
    parser.add_argument("-c","--create", type=str, help="create a new O2O job" )
    parser.add_argument("-1","--enable", type=str, help="enable the O2O job" )
    parser.add_argument("-0","--disable", type=str, help="disable the O2O job" )
   
    parser.add_argument("-t","--tag", type=str, help="the CondDB Tag name")
    parser.add_argument("-i","--interval", type=str, help="the chron job interval")
    parser.add_argument("-d","--dev", action="store_true", help="bookkeeping in dev database")
    parser.add_argument("-p","--private", action="store_true", help="bookkeeping in private database")
    parser.add_argument("-a","--auth", type=str,  help="path of the authentication file")
    args = parser.parse_args()  

    if not args.create and not args.enable and not args.disable:
        parser.error("Command not given. Possible choices: create, enable, disable")

    db_service = None
    if not args.private:
        if args.dev:
            db_service = o2olib.dev_db_service
        else:
            db_service = o2olib.prod_db_service
    mgr = o2olib.O2OJobMgr()
    ret = -1
    if mgr.connect( db_service, args.auth ):
        if args.create:
            if not args.tag:
                parser.error("Option 'tag' not provided.")
            if not args.interval:
                parser.error("Option 'interval' not provided.")
            print 'creating...'
            created = mgr.add( args.create, args.tag, args.interval, True )
            return created
        if args.enable:
            mgr.set( args.enable, True )
        if args.disable:
            mgr.set( args.disable, False )
        ret = 0
    return ret

if __name__ == '__main__':

    sys.exit(main(sys.argv))
