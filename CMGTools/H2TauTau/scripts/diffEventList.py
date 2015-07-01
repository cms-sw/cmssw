#!/bin/env python

import sys
from ROOT import TFile, TTree

import FWCore.ParameterSet.Config as cms


def readFile(fileName, eventCol=2):
    file = open(fileName)
    entries = []
    events = []
    nentries = None
    for line in file:
        line = line.lstrip().rstrip()
        spl = line.split('*')
        if len(spl)>1:
            try:
                entry = int( spl[1] )
                event = int( spl[eventCol] )
                entries.append(entry)
                events.append( (event,line) )
            except IndexError:
                pass
            except ValueError:
                pass
        else:
            spl = line.split()
            if len(spl)==4 and spl[0]=='==>':
                nentries = int(spl[1])
    if nentries: assert(nentries==len(entries))
    return entries, events


def printEvents(eventsToPrint, evlines):
    if len(eventsToPrint)==0:
        print 'None!'
    evRange = cms.VEventRange() 
    for ev, line in evlines:
        if ev in eventsToPrint:
            print line
    


def buildVEventRange(eventsToPrint, evlines):
    evRange = cms.VEventRange() 
    for ev, line in evlines:
        if ev in eventsToPrint:
            spl = line.split('*')
            # print line
            # print spl
            run = spl[2].rstrip().lstrip()
            evt = spl[4].rstrip().lstrip()
            evRange.append('{run}:{evt}'.format(run=run, evt=evt))
    return evRange
    
def buildCutStr(eventsToPrint, evlines, runName='run', evtName='evt'):
    cutStrElems = []
    for ev, line in evlines:
        if ev in eventsToPrint:
            spl = line.split('*')
            run = spl[2].rstrip().lstrip()
            evt = spl[4].rstrip().lstrip()
            cutStrElems.append('({runName}=={run} && {evtName}=={evt})'.format(
                runName = runName,
                evtName = evtName,
                run = run,
                evt = evt))
    return ' || '.join(cutStrElems)


from optparse import OptionParser

parser = OptionParser()


parser.add_option("-c", "--eventcol", dest="eventCol",
                  default=4,
                  help='column containing event number')
parser.add_option("-d", "--details", dest="details",
                  default=False,
                  action='store_true',
                  help='print VEventRange and root cutstring for further selection')

(options,args) = parser.parse_args()
if len(args)!=2:
    print 'provide 2 input text files'
    sys.exit(1)
eventCol = int(options.eventCol)
file1 = args[0]
file2 = args[1]
en1, ev1 = readFile( file1, eventCol )
en2, ev2 = readFile( file2, eventCol )

sev1 = set([ev for ev, line in ev1])
sev2 = set([ev for ev, line in ev2])

sep_line = '-'*100
import pprint
print file1, '-', file2
print sep_line
# pprint.pprint( sev1 - sev2 )
printEvents(sev1-sev2, ev1)
# import pdb; pdb.set_trace()
if options.details:
    print 
    print buildVEventRange(sev1-sev2, ev1)
    print
    print buildCutStr(sev1-sev2, ev1)

print
print
print file2, '-', file1
print sep_line
# pprint.pprint( sev2 - sev1 )
printEvents(sev2-sev1, ev2)

if options.details:
    print 
    print buildVEventRange(sev2-sev1, ev2)
    print
    print buildCutStr(sev2-sev1, ev2)
