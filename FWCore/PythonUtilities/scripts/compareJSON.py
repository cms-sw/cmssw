#!/usr/bin/env python

import sys
import optparse
from FWCore.PythonUtilities.LumiList import LumiList


if __name__ == '__main__':
    
    parser = optparse.OptionParser ("Usage: %prog --command [--options] alpha.json beta.json [output.json]")
    # required parameters
    cmdGroup = optparse.OptionGroup (parser, "Command Options ")
    cmdGroup.add_option ('--and', dest='command', action='store_const',
                         const='and', 
                         help = '"and" (i.e., take intersection) of two files')
    cmdGroup.add_option ('--or', dest='command', action='store_const',
                         const='or',
                         help = '"or" (i.e., take union) of two files')
    cmdGroup.add_option ('--sub', dest='command', action='store_const',
                         const='sub',
                         help = '"subtraction" (i.e., lumi sections in alpha not in beta) of two files')
    parser.add_option_group (cmdGroup)
    (options, args) = parser.parse_args()
    if len (args) < 2 or len (args) > 3:
        raise RuntimeError, "Two input filenames with one optional output filename must be provided."
    if not options.command:
        raise RunetimeError, "Exactly one command option must be specified"

    alphaList = LumiList (filename = args[0])  # Read in first JSON file
    betaList  = LumiList (filename = args[1])  # Read in first JSON file
    
    # print J1List
    OutList={}

    if options.command == 'and':
        outputList = alphaList & betaList

    if options.command == 'or':
        outputList = alphaList | betaList

    if options.command == 'sub':
        outputList = alphaList - betaList

    if len (args) >= 3:
        outputList.writeJSON (args[2])
    else:
        # print to screen
        print outputList

