#!/usr/bin/env python3

from __future__ import print_function
import sys
import optparse
import re
from FWCore.PythonUtilities.LumiList import LumiList


if __name__ == '__main__':
    
    parser = optparse.OptionParser ("Usage: %prog input.json input.csv output.csv")
    parser.add_option ('--output', dest='output', type='string',
                       help='Save output to file OUTPUT')
    parser.add_option ('--runIndex', dest='runIndex', type='int',
                       default = 0,
                       help='column to be converted to run number (default %default)')
    parser.add_option ('--lumiIndex', dest='lumiIndex', type='int',
                       default = 1,
                       help='column to be converted to lumi section number (default %default)')
    parser.add_option ('--noWarnings', dest='noWarnings', action='store_true',
                       help='do not print warnings about lines not matching run, lumi numbers')
    # required parameters
    (options, args) = parser.parse_args()
    if len (args) != 3:
        raise RuntimeError("Must provide an input JSON file, an input CSV file, and an output CSV file")

    sepRE = re.compile (r'[\s,;:]+')
    runLumiDict = {}
    jsonList = LumiList (args[0])
    source = open (args[1], 'r')
    target = open (args[2], 'w')
    runIndex, lumiIndex = options.runIndex, options.lumiIndex
    minPieces = max (runIndex, lumiIndex) + 1
    for line in source:
        copy = line.strip()
        pieces = sepRE.split (copy.strip())
        if len (pieces) < minPieces:
            if not options.noWarnings:
                print("Saving line '%s' since no identifiable run and lumi info" \
                      % copy)
            target.write (line)
            continue
        try:
            run, lumi = int( pieces[runIndex] ), int( pieces[lumiIndex] )
        except:
            if not options.noWarnings:
                print("Saving line '%s' since no identifiable run,lumi info" \
                      % copy)
            target.write (line)
            continue
        # OK.  We recognize this line as containing a valid run and
        # lumi number.  Is it part of the JSON file we provided?
        if (run, lumi) in jsonList:
            # Yes, it is.  Take it!
            target.write (line)
            
    source.close()
    target.close()
