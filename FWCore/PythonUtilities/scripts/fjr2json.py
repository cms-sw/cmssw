#! /usr/bin/env python

from FWCore.PythonUtilities.XML2Python import xml2obj
from FWCore.PythonUtilities.LumiList   import LumiList
from pprint import pprint
import optparse
import sys


if __name__ == '__main__':
    
    parser = optparse.OptionParser ("Usage: %prog [--options] job1.fjr [job2.fjr...]")
    parser.add_option ('--output', dest='output', type='string',
                       help='Save output to file OUTPUT')
    (options, args) = parser.parse_args()
    if not args:
        raise RuntimeError, "Must provide at least one input file"

    runsLumisDict = {}
    for fjr in args:
        try:
            obj = xml2obj (filename=fjr)
        except:
            print "'%s' is not an framework job report.  Skipping." % fjr
            continue
        for inputFile in obj.InputFile:
            try:
                runList = inputFile.Runs.Run
            except:
                try:
                    print "'%s' in '%s' contains no runs.  Skipping." % \
                          (inputFile.PFN, fjr)
                except:
                    print "Some run in '%s' contains no runs.  Skipping." % \
                          fjr
                continue
            for run in runList:
                runNumber = int (run.ID)
                runList = runsLumisDict.setdefault (runNumber, [])
                for lumiPiece in run.LumiSection:
                    lumi = int (lumiPiece.ID)
                    runList.append (lumi)

    jsonList = LumiList (runsAndLumis = runsLumisDict)
    if options.output:
        jsonList.writeJSON (options.output)
    else:
        print jsonList
                
