#! /usr/bin/env python3

from __future__ import print_function
from FWCore.PythonUtilities.XML2Python import xml2obj
from FWCore.PythonUtilities.LumiList   import LumiList
from pprint import pprint

import ast
import optparse
import sys

import six

if __name__ == '__main__':

    parser = optparse.OptionParser ("Usage: %prog [--options] job1.fjr [job2.fjr...]")
    parser.add_option ('--output', dest='output', type='string',
                       help='Save output to file OUTPUT')
    (options, args) = parser.parse_args()
    if not args:
        raise RuntimeError("Must provide at least one input file")

    runsLumisDict = {}
    for fjr in args:
        try:
            obj = xml2obj (filename=fjr)
        except:
            print("'%s' is not an framework job report.  Skipping." % fjr)
            continue
        for inputFile in obj.InputFile:
            try: # Regular XML version, assume only one of these
                runObjects = inputFile.Runs.Run
                for run in runObjects:
                    runNumber = int (run.ID)
                    runList = runsLumisDict.setdefault (runNumber, [])
                    for lumiPiece in run.LumiSection:
                        lumi = int (lumiPiece.ID)
                        runList.append (lumi)
            except:
                try: # JSON-like version in CRAB XML files, runObjects is usually a list
                    if isinstance(inputFile.Runs, str):
                        runObjects = [inputFile.Runs]
                    else:
                        runObjects = inputFile.Runs

                    for runObject in runObjects:
                        try:
                            runs = ast.literal_eval(runObject)
                            for (run, lumis) in six.iteritems(runs):
                                runList = runsLumisDict.setdefault (int(run), [])
                                runList.extend(lumis)
                        except ValueError: # Old style handled above
                            pass
                except:
                    print("Run missing in '%s'.  Skipping." % fjr)
                continue

    jsonList = LumiList (runsAndLumis = runsLumisDict)
    if options.output:
        jsonList.writeJSON (options.output)
    else:
        print(jsonList)

