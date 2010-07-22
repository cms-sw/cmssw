#!/usr/bin/env python

import sys
import optparse
from FWCore.PythonUtilities.LumiList import LumiList


if __name__ == '__main__':
    
    parser = optparse.OptionParser ("Usage: %prog alpha.json")
    # required parameters
    (options, args) = parser.parse_args()
    if len (args) != 1:
        raise RuntimeError, "Must provide exactly one input file"

    alphaList = LumiList (filename = args[0])  # Read in first  JSON file
    print alphaList
