#! /usr/bin/env python

from FWCore.PythonUtilities.LumiList   import LumiList
import optparse


if __name__ == '__main__':
    
    parser = \optparse.OptionParser ("Usage: %prog [--options] edm1.root [edm2.root...]",
                                    description='Runs over input EDM files and prints out a list of contained lumi sections')
    parser.add_option ('--output', dest='output', type='string',
                       help='Save output to file OUTPUT')
    (options, args) = parser.parse_args()
    # put this here after parsing the arguments since ROOT likes to
    # grab command line arguments even when it shouldn't.
    from DataFormats.FWLite import Lumis
    if not args:
        raise RuntimeError, "Must provide at least one input file"

    runsLumisDict = {}
    lumis = Lumis (args)
    for lum in lumis:
        runList = runsLumisDict.setdefault (lum.aux().run(), [])
        runList.append( lum.aux().id().luminosityBlock() )

    jsonList = LumiList (runsAndLumis = runsLumisDict)
    if options.output:
        jsonList.writeJSON (options.output)
    else:
        print jsonList
