#!/usr/bin/env python3

from __future__ import print_function
import sys
import optparse
import re
from FWCore.PythonUtilities.LumiList import LumiList


def filterRuns (lumiList, minRun, maxRun):
    allRuns = lumiList.getRuns()
    runsToRemove = []
    for run in allRuns:
        if minRun and int(run) < minRun:
            runsToRemove.append (run)
        if maxRun and int(run) > maxRun:
            runsToRemove.append (run)
    lumiList.removeRuns (runsToRemove)
    


if __name__ == '__main__':
    
    parser = optparse.OptionParser ("Usage: %prog alpha1.json [alpha2.json:142300-145900]")
    parser.add_option ('--output', dest='output', type='string',
                       help='Save output to file OUTPUT')
    # required parameters
    (options, args) = parser.parse_args()
    if not len (args):
        raise RuntimeError("Must provide at least one input file")

    minMaxRE = re.compile (r'(\S+):(\d+)-(\d*)')

    finalList = LumiList()
    for filename in args:
        minRun = maxRun = 0
        match = minMaxRE.search (filename)
        if match:
            filename   =      match.group(1)
            minRun     = int( match.group(2) )
            try:
                maxRun = int( match.group(3) )
            except:
                pass
            if maxRun and minRun > maxRun:
                raise RuntimeError("Minimum value (%d) is greater than maximum value (%d) for file '%s'" % (minRun, maxRun, filename))
        localList = LumiList (filename = filename)
        filterRuns (localList, minRun, maxRun)
        finalList = finalList | localList

    if options.output:
        finalList.writeJSON (options.output)
    else:
        print(finalList)
