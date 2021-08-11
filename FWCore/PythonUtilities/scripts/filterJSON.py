#!/usr/bin/env python3

from __future__ import print_function
import sys
import optparse
import re
from FWCore.PythonUtilities.LumiList import LumiList


if __name__ == '__main__':
    
    parser = optparse.OptionParser ("Usage: %prog alpha.json")
    parser.add_option ('--max', dest='max', type='int', default=0,
                       help='maximum run to keep in output')
    parser.add_option ('--min', dest='min', type='int', default=0,
                       help='minimum run to keep in output')
    parser.add_option ('--runs', dest='runs', type='string',
                       action='append', default = [],
                       help='runs to remove from JSON file')
    parser.add_option ('--output', dest='output', type='string',
                       help='Save output to file OUTPUT')
    # required parameters
    (options, args) = parser.parse_args()
    if len (args) != 1:
        raise RuntimeError("Must provide exactly one input file")

    if options.min and options.max and options.min > options.max:
        raise RuntimeError("Minimum value (%d) is greater than maximum value (%d)" % (options.min, options.max))

    commaRE = re.compile (r',')
    runsToRemove = []
    for chunk in options.runs:
        runs = commaRE.split (chunk)
        runsToRemove.extend (runs)

    alphaList = LumiList (filename = args[0])  # Read in first  JSON file
    allRuns = alphaList.getRuns()
    for run in allRuns:
        if options.min and int(run) < options.min:
            runsToRemove.append (run)
        if options.max and int(run) > options.max:
            runsToRemove.append (run)

    alphaList.removeRuns (runsToRemove)

    if options.output:
        alphaList.writeJSON (options.output)
    else:
        print(alphaList)
