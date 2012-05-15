#! /usr/bin/env python

import optparse
import re
from pprint import pprint

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
        raise RuntimeError, "You have not provided any files"

    events = Events (listOfFiles)

    handle = Handle('PileupSummaryInfo')
    label  = ('addPileupInfo')

    ROOT.gROOT.SetBatch()        # don't pop up canvases

    # loop over events
    countDict = {}
    total = 0.
    for event in events:
        event.getByLabel (label, handle)
        pileup = handle.product()
        num = pileup.getPU_NumInteractions()
        total += 1
        if not countDict.has_key (num):
            countDict[num] = 1
        else:
            countDict[num] += 1

    print "total", int(total), "\ncounts:"
    pprint (countDict, width=1)
    print "normalized:"

    renormDict = {}
    for key, count in countDict.iteritems():
        renormDict[key] = count / total
    pprint (renormDict)
    
