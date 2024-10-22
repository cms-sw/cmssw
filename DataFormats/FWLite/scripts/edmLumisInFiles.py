#! /usr/bin/env python3

from FWCore.PythonUtilities.LumiList import LumiList
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser(description='Runs over input EDM files and prints out a list of contained lumi sections')
    parser.add_argument('--intLumi', dest='intLumi', action='store_true',
                        help='print out total recorded and delivered integrated luminosity')
    parser.add_argument('--output', dest='output', type=str,
                        help='save lumi sections output to file OUTPUT')
    parser.add_argument("edm", metavar="edm.root", type=str, nargs='+')
    options = parser.parse_args()
    # put this here after parsing the arguments since ROOT likes to
    # grab command line arguments even when it shouldn't.
    from DataFormats.FWLite import Lumis, Handle

    # do we want to get the luminosity summary?
    if options.intLumi:
        handle = Handle ('LumiSummary')
        label  = ('lumiProducer')
    else:
        handle, lable = None, None

    runsLumisDict = {}
    lumis = Lumis (options.edm)
    delivered = recorded = 0
    for lum in lumis:
        runList = runsLumisDict.setdefault (lum.aux().run(), [])
        runList.append( lum.aux().id().luminosityBlock() )
        # get the summary and keep track of the totals
        if options.intLumi:
            lum.getByLabel (label, handle)
            summary = handle.product()
            delivered += summary.avgInsDelLumi()
            recorded  += summary.avgInsRecLumi()

    # print out lumi sections in JSON format
    jsonList = LumiList (runsAndLumis = runsLumisDict)
    if options.output:
        jsonList.writeJSON (options.output)
    else:
        print(jsonList)

    # print out integrated luminosity numbers if requested
    if options.intLumi:
        print("\nNote: These numbers should be considered approximate.  For official numbers, please use lumiCalc.py")
        print("delivered %.1f mb,  recorded %.1f mb" % \
              (delivered, recorded))
