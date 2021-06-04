#! /usr/bin/env python3

from __future__ import print_function
import optparse
import re
from pprint import pprint
import six

commentRE = re.compile (r'#.*$')

if __name__ == "__main__":
    parser = optparse.OptionParser ("Usage: %prog file1.root [file2.root...]")
    parser.add_option ('--loadFromFile', dest='loadFromFile', default=[],
                       type='string',
                       action='append', 
                       help="Name of text file containing filenames" )
    parser.add_option ('--prefix', dest='prefix', type='string',
                       default='',
                       help="Prefix to add to files" )

    parser.add_option ('--bx', dest='bx', type='int',
                       default='0',
                       help="Bunch crossing to check (0 = in-time)" )
    (options, args) = parser.parse_args()
    import ROOT # stupid ROOT takes the arugments error
    from DataFormats.FWLite import Events, Handle

    listOfFiles = args[:]
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
    for key, count in six.iteritems(countDict):
        renormDict[key] = count / total
    pprint (renormDict)
    
