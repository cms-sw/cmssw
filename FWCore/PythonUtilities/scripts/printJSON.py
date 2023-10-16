#!/usr/bin/env python3

from __future__ import print_function
import sys
from argparse import ArgumentParser
from FWCore.PythonUtilities.LumiList import LumiList

if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('--range', dest='range', default=False, action='store_true',
                         help='Print out run range only')
    parser.add_argument("alpha_json", metavar="alpha.json", type=str)
    options = parser.parse_args()

    alphaList = LumiList (filename = options.alpha_json) # Read in first JSON file
    if options.range:
        keys = alphaList.compactList.keys()
        minRun = min (keys)
        maxRun = max (keys)
        print("runs %s - %s" % (minRun, maxRun))
        sys.exit()
    print(alphaList)
