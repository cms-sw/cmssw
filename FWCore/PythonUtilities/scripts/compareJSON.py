#!/usr/bin/env python3

from __future__ import print_function
import sys
from argparse import ArgumentParser
from FWCore.PythonUtilities.LumiList import LumiList


if __name__ == '__main__':
    
    parser = ArgumentParser()
    # required parameters
    cmdGroupTitle = parser.add_argument_group("Command Options")
    cmdGroup = cmdGroupTitle.add_mutually_exclusive_group(required=True)
    cmdGroup.add_argument('--and', dest='command', action='store_const',
                          const='and',
                          help = '"and" (i.e., take intersection) of two files')
    cmdGroup.add_argument('--or', dest='command', action='store_const',
                          const='or',
                          help = '"or" (i.e., take union) of two files')
    cmdGroup.add_argument('--sub', dest='command', action='store_const',
                          const='sub',
                          help = '"subtraction" (i.e., lumi sections in alpha not in beta) of two files')
    cmdGroup.add_argument('--diff', dest='command', action='store_const',
                          const='diff',
                          help = '"All differences" (i.e., alpha - beta AND beta - alpha) of two files. Output will only be to screen (not proper JSON format).')
    parser.add_argument("alpha", metavar="alpha.json", type=str)
    parser.add_argument("beta", metavar="beta.json", type=str)
    parser.add_argument("output", metavar="output.json", type=str, nargs='?', default=None)
    options = parser.parse_args()
    if not options.command:
        parser.error("Exactly one command option must be specified")

    alphaList = LumiList (filename = options.alpha)  # Read in first  JSON file
    betaList  = LumiList (filename = options.beta)  # Read in second JSON file

    ##################
    ## Diff Command ##
    ##################
    if options.command == 'diff':
        if options.output is not None:
            raise RuntimeError("Can not output to file with '--diff' option.  The output is not standard JSON.")
        firstOnly  = alphaList - betaList
        secondOnly = betaList  - alphaList
        if not firstOnly and not secondOnly:
            print("Files '%s' and '%s' are the same." % (options.alpha, options.beta))
            sys.exit()
        print("'%s'-only lumis:" % options.alpha)
        if firstOnly:
            print(firstOnly)
        else:
            print("None")
        print("\n'%s'-only lumis:" % options.beta)
        if secondOnly:
            print(secondOnly)
        else:
            print("None")
        sys.exit()
    
    ########################
    ## All other commands ##
    ########################
    if options.command == 'and':
        outputList = alphaList & betaList

    if options.command == 'or':
        outputList = alphaList | betaList

    if options.command == 'sub':
        outputList = alphaList - betaList

    if options.output is not None:
        outputList.writeJSON(options.output)
    else:
        # print to screen
        print(outputList)

