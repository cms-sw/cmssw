#! /usr/bin/env python3

from argparse import ArgumentParser
import re
from pprint import pprint

commentRE = re.compile (r'#.*$')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--loadFromFile', dest='loadFromFile', default=[],
                        type=str,
                        action='append', 
                        help="Name of text file containing filenames" )
    parser.add_argument('--prefix', dest='prefix', type=str,
                        default='',
                        help="Prefix to add to files" )
    parser.add_argument('--bx', dest='bx', type=int,
                        default='0',
                        help="Bunch crossing to check (0 = in-time)" )
    parser.add_argument("file", metavar="file.root", type=str, nargs='*')
    options = parser.parse_args()
    import ROOT # stupid ROOT takes the arguments error
    from DataFormats.FWLite import Events, Handle

    listOfFiles = options.file
    for filename in options.loadFromFile:
        source = open (filename, 'r')
        for line in source:            
            line = commentRE.sub ('', line).strip() # get rid of comments
            if not line:
                # don't bother with blank lines
                continue
            listOfFiles.append (line)
        source.close()
    if options.prefix:
        oldList = listOfFiles
        listOfFiles = []
        for name in oldList:
            listOfFiles.append( options.prefix + name )

    if not listOfFiles:
        raise RuntimeError("You have not provided any files")

    events = Events (listOfFiles)

    handle = Handle('vector<PileupSummaryInfo>')
    label  = ('addPileupInfo')

    ROOT.gROOT.SetBatch()        # don't pop up canvases

    # loop over events
    countDict = {}
    total = 0.
    for event in events:
        event.getByLabel (label, handle)
        pileups = handle.product()
        for pileup in pileups:
            if pileup.getBunchCrossing() == options.bx:
                break
            if pileup == pileups[-1] and len(pileups)>1 :
                raise RuntimeError("Requested BX not found in file")

        num = pileup.getPU_NumInteractions()
        total += 1
        if num not in countDict:
            countDict[num] = 1
        else:
            countDict[num] += 1

    print("total", int(total), "\ncounts:")
    pprint (countDict, width=1)
    print("normalized:")

    renormDict = {}
    for key, count in countDict.items():
        renormDict[key] = count / total
    pprint (renormDict)
    
