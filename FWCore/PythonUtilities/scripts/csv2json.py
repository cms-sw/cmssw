#!/usr/bin/env python3

from __future__ import print_function
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
    parser.add_argument("input", metavar="input.csv", type=str)
    options = parser.parse_args()

    sepRE = re.compile (r'[\s,;:]+')
    runLumiDict = {}    
    events = open (options.input, 'r')
    runIndex, lumiIndex = options.runIndex, options.lumiIndex
    minPieces = max (runIndex, lumiIndex) + 1
    for line in events:
        pieces = sepRE.split (line.strip())
        if len (pieces) < minPieces:
            continue
        try:
            run, lumi = int( pieces[runIndex] ), int( pieces[lumiIndex] )
        except:
            continue
        runLumiDict.setdefault (run, []).append (lumi)
    jsonList = LumiList (runsAndLumis = runLumiDict) 
    if options.output:
        jsonList.writeJSON (options.output)
    else:
        print(jsonList)
