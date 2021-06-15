#!/usr/bin/env python3

from __future__ import print_function
import sys
import optparse
from FWCore.PythonUtilities.LumiList import LumiList


if __name__ == '__main__':
    
    parser = optparse.OptionParser ("Usage: %prog alpha.json")
    parser.add_option ('--range', dest='range', action='store_true',
                       help='Print out run range only')
    # required parameters
    (options, args) = parser.parse_args()
    if len (args) != 1:
        raise RuntimeError("Must provide exactly one input file")

    alphaList = LumiList (filename = args[0])  # Read in first  JSON file
    if options.range:
        keys = alphaList.compactList.keys()
        minRun = min (keys)
        maxRun = max (keys)
        print("runs %s - %s" % (minRun, maxRun))
        sys.exit()
    print(alphaList)
