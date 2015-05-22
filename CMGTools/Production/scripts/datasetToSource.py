#!/usr/bin/env python
import os 
import fnmatch
from CMGTools.Production.datasetToSource import *

if __name__ == '__main__':
    import sys,pprint

    from optparse import OptionParser

    parser = OptionParser()
    parser.usage = "%prog [options] <dataset>\nPrints information on a sample."
    parser.add_option("-w", "--wildcard", dest="wildcard", default='*tree*root',help='UNIX style wildcard for root file printout')
    parser.add_option("-u", "--user", dest="user", default=os.environ['USER'],help='user owning the dataset')

    (options,args) = parser.parse_args()

    if len(args)!=1:
        parser.print_help()
        sys.exit(1)

    user = options.user
    dataset = args[0]
    pattern = fnmatch.translate( options.wildcard )
    
    source = datasetToSource( user, dataset, pattern )
    dump = 'source = '
    dump += source.dumpPython()
    dump = dump.replace("'/store","\n'/store")
    print 'import FWCore.ParameterSet.Config as cms'
    print dump
