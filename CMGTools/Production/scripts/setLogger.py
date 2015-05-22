#!/usr/bin/env python

import os

from CMGTools.Production.logger import *

if __name__ == '__main__':

    from optparse import OptionParser

    parser = OptionParser()
    
    parser.usage = """setLogger.py 
    """
    parser.add_option("-n", "--number_of_jobs", dest="number_of_jobs",
                      help="Specify original number of jobs",
                      default=0)


    (options,args) = parser.parse_args()

    if len(args)!=0:
        parser.print_help()
        sys.exit(1)

    dir = 'Logger'
    if not os.path.isdir(dir):
        os.mkdir(dir)
    log = logger(dir)
    log.logCMSSW()
    log.logJobs( int(options.number_of_jobs) )
    
