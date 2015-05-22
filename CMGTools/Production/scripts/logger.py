#!/usr/bin/env python

from CMGTools.Production.logger import *

if __name__ == '__main__':

    from optparse import OptionParser

    parser = OptionParser()
    
    parser.usage = """logger.py <dir or castortgz>
Get information on the software that was used to process a dataset.

Example:
logger.py /store/cmst3/user/lucieg/CMG/DoubleMu/Run2011A-May10ReReco-v1/AOD/PAT_CMG/Logger.tgz
more Logger/*
    """

    (options,args) = parser.parse_args()

    if len(args)!=1:
        parser.print_help()
        sys.exit(1)

    dirOrFile = args[0]

    try:
        log = logger(dirOrFile)
        # log.logCMSSW()

        if log.dirLocal == None:
            log.stageIn()
            
        # log.addFile('patTuple_PATandPF2PAT_RecoJets_cfg.py')
        log.stageOut('/store/cmst3/user/cbern/Test')
        # log.dump()
    except ValueError as err:
        print err, '. Exit!'
        sys.exit(1)
