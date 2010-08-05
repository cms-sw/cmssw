#! /usr/bin/env python

from FWCore.PythonUtilities.XML2Python import xml2obj
from FWCore.PythonUtilities.LumiList   import LumiList
from pprint import pprint
import optparse

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
            print "'%s' is not an framework job report.  Skipping."
            continue
        for inputFile in obj.InputFile:
            runNumber = int (inputFile.Runs.Run.ID)
            runList = runsLumisDict.setdefault (runNumber, [])
            for lumiPiece in inputFile.Runs.Run.LumiSection:
                lumi = int (lumiPiece.ID)
                runList.append (lumi)

    jsonList = LumiList (runsAndLumis = runsLumisDict)
    if options.output:
        target = open (options.output, 'w')
        target.write ("%s\n" % jsonList)
        target.close()
    else:
        print jsonList
                
