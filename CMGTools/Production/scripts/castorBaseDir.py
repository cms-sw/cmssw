#!/usr/bin/env python
import CMGTools.Production.eostools as castortools
from CMGTools.Production.castorBaseDir import castorBaseDir

if __name__ == '__main__':
    import sys
    from optparse import OptionParser

    parser = OptionParser()
    parser.usage = "%prog <user>\nPrints the castor base directory of a given user."
    parser.add_option("-c", "--castorpath", action="store_true",
                      dest="castorpath",
                      help="Print full castor path. Otherwise print LFN, starting by /store",
                      default=False)

    (options,args) = parser.parse_args()

    if len(args)!=1:
        parser.print_help()
        sys.exit(1)

    user = args[0]
    
    dir = castorBaseDir(user)
    if options.castorpath:
        dir = castortools.lfnToCastor( dir )
    print dir
