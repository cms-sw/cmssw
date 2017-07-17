#!/usr/bin/env python

import os
import pprint
import fnmatch
from PhysicsTools.HeppyCore.utils.dataset import createDataset

if __name__ == '__main__':

    import sys
    from optparse import OptionParser
    import pprint
    
    parser = OptionParser()
    parser.usage = "%prog [options] <dataset>\nPrints information on a sample."
    parser.add_option("-w", "--wildcard", dest="wildcard", default='tree*root',help='A UNIX style wilfcard for root file printout')
    parser.add_option("-u", "--user", dest="user", default=os.environ.get('USER', None),help='user owning the dataset.\nInstead of the username, give "LOCAL" to read datasets in a standard unix filesystem, and "CMS" to read official CMS datasets present at CERN.')
    parser.add_option("-b", "--basedir", dest="basedir", default=os.environ.get('CMGLOCALBASEDIR',None),help='in case -u LOCAL is specified, this option allows to specify the local base directory containing the dataset. default is CMGLOCALBASEDIR')
    parser.add_option("-a", "--abspath", dest="abspath",
                      action = 'store_true',
                      default=False,
                      help='print absolute path')
    parser.add_option("-n", "--noinfo", dest="noinfo",
                      action = 'store_true',
                      default=False,
                      help='do not print additional info (file size and status)')
    parser.add_option("-r", "--report", dest="report",
                      action = 'store_true',
                      default=False,
                      help='Print edmIntegrityCheck report')
    parser.add_option("-c", "--readcache", dest="readcache",
                      action = 'store_true',
                      default=False,
                      help='Read from the cache.')
    parser.add_option("--min-run", dest="min_run", default=-1, type=int, help='When querying DBS, require runs >= than this run')
    parser.add_option("--max-run", dest="max_run", default=-1, type=int, help='When querying DBS, require runs <= than this run')

    (options,args) = parser.parse_args()

    if len(args)!=1:
        parser.print_help()
        sys.exit(1)

    user = options.user
    name = args[0]
    info = not options.noinfo

    run_range = (options.min_run,options.max_run)
    data = createDataset(user, name,
                         fnmatch.translate( options.wildcard ),
                         options.readcache,
                         options.basedir,
                         run_range=run_range)
    data.printInfo()
    data.printFiles(abspath = options.abspath,
                    info = info)
    pprint.pprint( data.filesAndSizes )
    if options.report:
        pprint.pprint( data.report )


