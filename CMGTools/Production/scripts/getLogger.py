#!/usr/bin/env python

from CMGTools.Production.logger import *

if __name__ == '__main__':

    from optparse import OptionParser

    parser = OptionParser()
    
    parser.usage = """logger.py <castortgz>
Get information on the software that was used to process a dataset.

Example:
logger.py /store/cmst3/user/lucieg/CMG/DoubleMu/Run2011A-May10ReReco-v1/AOD/PAT_CMG/Logger.tgz
more Logger/*
    """

    (options,args) = parser.parse_args()

    if len(args)!=1:
        parser.print_help()
        sys.exit(1)

    tgzFile = args[0]

    log = logger(tgzFile)
    log.stageIn()
            
 
