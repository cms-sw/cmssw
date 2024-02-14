#!/usr/bin/env python3

import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import re
from FWCore.PythonUtilities.LumiList import LumiList

if __name__ == '__main__':
    
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output', dest='output', type=str,
                        help='Save output to file OUTPUT')
    parser.add_argument('--runIndex', dest='runIndex', type=int,
                        default = 0,
                        help='column to be converted to run number')
    parser.add_argument('--lumiIndex', dest='lumiIndex', type=int,
                        default = 1,
                        help='column to be converted to lumi section number')
    parser.add_argument('--noWarnings', dest='noWarnings', action='store_true',
                        default = False,
                        help='do not print warnings about lines not matching run, lumi numbers')
    parser.add_argument("input_json", metavar="input.json", type=str)
    parser.add_argument("input_csv", metavar="input.csv", type=str)
    parser.add_argument("output_csv", metavar="output.csv", type=str)
    options = parser.parse_args()

    sepRE = re.compile (r'[\s,;:]+')
    runLumiDict = {}
    jsonList = LumiList(options.input_json)
    source = open(options.input_csv, 'r')
    target = open(options.output_csv, 'w')
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
