#!/bin/env python

from CMGTools.Production.hadd import haddRec


if __name__ == '__main__':

    import sys
    from optparse import OptionParser

    parser = OptionParser()
    parser.usage = """
    %prog <outdir> <list of input directories>
    Like hadd, but works on directories.
    """

    (options,args) = parser.parse_args()

    if len(args)<3:
        print 'provide at least 2 directories to be added.'
        sys.exit(1)

    odir = args[0]
    idirs = args[1:]
    haddRec(odir, idirs)
