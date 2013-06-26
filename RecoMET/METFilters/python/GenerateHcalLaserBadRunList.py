#!/usr/bin/env python

import sys, os, string
import time

import FWCore.ParameterSet.Config as cms
from hcalLaserEventFilter_cfi import hcalLaserEventFilter

''' Program reads existing bad run/event list from hcalLaserEventFilter_cfi.py, and an (optional) new list from a text file.  If a text file is specified, this is assumed to be the desired new bad list, and its output will be sent to badEvents.py in the form needed by the cfi file.

If no text file is provided, then the current bad events in the .py file will be displayed.
'''


def MakePair(startlist):
    dict={}
    for i in range(0,len(startlist),2):
        key1=startlist[i]
        key2=startlist[i+1]
        runevent=(key1,key2)
        dict[runevent]="%s,%s,"%(key1,key2)
    return dict

def ReadNewList(newlist):
    ''' Read a new list of bad runs from an input file, and
        creates a new list of output keys for the bad run/events.
        '''
    outlist=[]
    for i in newlist:
        temp=string.strip(i)
        # Input is comma-separated
        if temp.find(",")>-1:
            temp=string.split(temp,",")
        # Input is space-separated
        else:
            temp=string.split(temp)

        # Case 1:  new input presented as "run lumi event" list
        if len(temp)==3:
            try:
                run=string.atoi(temp[0])
                evt=string.atoi(temp[2])
            except:
                print "Could not parse line '%s'"%i
        # Case 2:  new input presented as "run event" list
        elif len(temp)==2:
            try:
                run=string.atoi(temp[0])
                evt=string.atoi(temp[1])
            except:
                print "Could not parse line '%s'"%i
        else:
            print "Cannot parse line! ('%s')"%i
            continue
        outlist.append(run)
        outlist.append(evt)
    outDict=MakePair(outlist)
    return outDict


#######################################################

if __name__=="__main__":
    defaultList=hcalLaserEventFilter.BadRunEventNumbers
    defaultDict=MakePair(defaultList)
    keys=defaultDict.keys()
    keys.sort()
    if len(sys.argv)==1:
        print "Default bad (run,events) are:"
        for i in keys:
            print i
        print "\nA total of %i bad events"%len(keys)
        sys.exit()
    newlines=[]
    for i in sys.argv[1:]:
        if not os.path.isfile(i):
            print "Error, file '%s' does not exist"%i
            continue
        lines=open(i,'r').readlines()
        for i in lines:
            newlines.append(i)
    newBadDict=ReadNewList(newlines)
    # At some point, add ability to append/replace.
    # For now, new list is just the output from newBadDict

    newkeys=newBadDict.keys()
    newkeys.sort()
    notInOld={}
    notInNew={}

    out=open("badEvents.py",'w')
    
    thistime=time.time()
    thistime=time.strftime("%H:%M:%S %d %h %Y")
    out.write("# File last updated on %s\n"%thistime)
    out.write("# A total of %i bad events\n\n"%len(newkeys))

    out.write("badEvents=[\n")
    for i in newkeys:
        #print newBadDict[i]
        out.write("%s\n"%newBadDict[i])
        if i not in keys:
            notInOld[i]=newBadDict[i]
    out.write("]\n")
    out.close()
    
    
    for i in keys:
        if i not in newkeys:
            notInNew[i]=defaultDict[i]


    print "Total bad events in new file = ",len(newkeys)

    if len(notInOld.keys())>0:
        print
        print "A total of %i bad events found"%len(notInOld.keys())
        for k in notInOld.keys():
            print k

    if len(notInNew.keys())>0:
        print
        print "A total of %i events aren't in NEW list!"%len(notInNew.keys())
        for k in notInNew.keys():
            print k
