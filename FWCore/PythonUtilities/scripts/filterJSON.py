#!/usr/bin/env python3

from __future__ import print_function
import sys
from argparse import ArgumentParser
import re
from FWCore.PythonUtilities.LumiList import LumiList

if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('--max', dest='max', type=int, default=0,
                        help='maximum run to keep in output')
    parser.add_argument('--min', dest='min', type=int, default=0,
                        help='minimum run to keep in output')
    parser.add_argument('--runs', dest='runs', type=str,
                        action='append', default = [],
                        help='runs to remove from JSON file')
    parser.add_argument('--output', dest='output', type=str,
                        help='Save output to file OUTPUT')
    parser.add_argument("alpha", metavar="alpha.json", type=str)
    # required parameters
    options = parser.parse_args()

    if options.min and options.max and options.min > options.max:
        raise RuntimeError("Minimum value (%d) is greater than maximum value (%d)" % (options.min, options.max))

    commaRE = re.compile (r',')
    runsToRemove = []
    for chunk in options.runs:
        runs = commaRE.split (chunk)
        runsToRemove.extend (runs)

    alphaList = LumiList (filename = options.alpha)  # Read in first JSON file
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
