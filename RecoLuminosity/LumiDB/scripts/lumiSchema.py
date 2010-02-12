#!/usr/bin/env python
VERSION='1.00'
import os,sys
from RecoLuminosity.LumiDB import argparse 

def main():
    parser = argparse.ArgumentParser(prog=sys.argv[0],description="Lumi DB schema operations.")
    # add the arguments
    parser.add_argument('-c',dest='connect',action='store',required=True,help='connect string to lumiDB')
    parser.add_argument('-P',dest='authpath',action='store',help='path to authentication file')    
    parser.add_argument('action',choices=['create','drop'],help='action on the schema')
    parser.add_argument('--verbose',dest='verbose',action='store_true',help='verbose')
    # parse arguments
    args=parser.parse_args()
    if args.action == 'create':
        print 'about to create lumi db schema'
    if args.action == 'drop':
        print 'about to drop lumi db schema'
    if args.verbose :
        print 'verbose mode'
if __name__=='__main__':
    main()
    
