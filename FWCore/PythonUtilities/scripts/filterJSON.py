#!/usr/bin/env python

import sys
import optparse
from FWCore.PythonUtilities.LumiList import LumiList


if __name__ == '__main__':
    
    parser = optparse.OptionParser ("Usage: %prog alpha.json")
    parser.add_option ('--max', dest='max', type='int', default=0,
                       help='maximum run to keep in output')
    parser.add_option ('--min', dest='min', type='int', default=0,
                       help='minimum run to keep in output')
    parser.add_option ('--output', dest='output', type='string',
                       help='Save output to file OUTPUT')
    # required parameters
    (options, args) = parser.parse_args()
    if len (args) != 1:
        raise RuntimeError, "Must provide exactly one input file"

    if options.min and options.max and options.min > options.max:
        raise RuntimeError, "Minimum value (%d) is greater than maximum value (%d)" % (options.min, options.max)

    alphaList = LumiList (filename = args[0])  # Read in first  JSON file
    newDict = {}
    for run, lumiArray in alphaList.compactList.iteritems():
        if options.min and int(run) < options.min:
            continue
        if options.max and int(run) > options.max:
            continue
        newDict[run] = lumiArray

    outputList = LumiList (compactList = newDict)
    if options.output:
        outputList.writeJSON (options.output)
    else:
        print outputList
