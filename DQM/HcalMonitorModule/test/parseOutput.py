#!/usr/bin/env python
'''
parseOutput.py

by Jeff Temple (jtemple@fnal.gov)

Code parses timing output from DQM to calculate avg time/event of each DQM subprocess.
To run, set process.hcalMonitor.showTiming = True in your cfg file, then
> cmsRun [your cfg file]  > timing.txt
> python parseOutput.py timing.txt

'''

import sys
import os
import string
from math import log

def computeAvg(mylist,dump=False):
    ''' Compute the average value of a list of numbers. '''
    avg=0
    if (len(mylist)==0):
        return avg
    if (dump==True):
        print "Length of list = ",len(mylist)
    count=0
    for i in mylist:
        count=count+1
        avg=avg+i
        if (dump==True):
            print "i = ",i,"  sum = ",avg," avg = ",(avg/count)
    avg=avg/len(mylist)
    return avg


myfile=sys.argv[1:]
print "\n\n"

# Run over all .txt files containing timing info
for F in myfile:
    print "Results for file %s:"%F
    print "------- --- ---- --------"

    mylines=open(F,'r').readlines()

    myTimers={} # dictionary of all TIMER objects
    # format of myTimers is key:[list], where key=name string, and [list]= list of time values for the given name

    for i in mylines:
        # timer objects must all begin with "TIMER::"
        if string.find(i,"TIMER::")>-1:
            temp=string.strip(i)
            # timer objects must also have an arrow (->) separating name from value
            temp=string.split(temp,"->")
            #print "..%s.."%temp[1]
            val=string.atof(temp[1])
            # Strip off "TIMER::" prefix when getting key name
            key=string.split(temp[0],"::")[1]
            key=string.strip(key)
            
            if key not in myTimers.keys():
                myTimers[key]=[val]
            else:
                myTimers[key].append(val)

    # Sort keys by name
    mykeys=myTimers.keys()
    mykeys.sort()
    
    myTime={}
    mySubTime={}
    sum=0
    sumsub=0

    # break timer keys into tasks and subtasks
    # (subtasks begin with "Hcal")

    dump=False
    for i in mykeys:
        #if (i=="HcalDeadCellMonitor PROCESSEVENT_DIGI"):
        #    dump=True
        #else:
        #    dump=False
        temp=computeAvg(myTimers[i],dump)
        #if (i=="HcalDeadCellMonitor PROCESSEVENT_DIGI"):
        #    print "AVG = ",i,temp
        if i[0:4]=="Hcal":
            mySubTime[temp]=i
            sumsub=sumsub+temp
        else:
            myTime[temp]=i
            sum=sum+temp


    # Print out timing info for overall tasks
    mykeys=myTime.keys()
    mykeys.sort()
    mykeys.reverse()

    maxevts=0
    for i in mykeys:
        if len(myTimers[myTime[i]])>maxevts:
            maxevts=len(myTimers[myTime[i]])
    maxdigits=int(log(maxevts,10)+.00001)

    newdict={}
    for i in mykeys:
        newdict[i*len(myTimers[myTime[i]])/maxevts]=i

    evtkeys=newdict.keys()
    evtkeys.sort()
    evtkeys.reverse()

    sum=0
    #for i in mykeys:
    for j in evtkeys:
        i=newdict[j]
        temp="%.5f : "%(i*len(myTimers[myTime[i]])/maxevts)
        temp=temp+"  %s"%(myTime[i])
        while (len(temp)<40):
            temp=temp+" "
        temp=temp+"   (%5i events -- time/event is %.5f)"%(len(myTimers[myTime[i]]),i)
        print temp
        sum=sum+i*len(myTimers[myTime[i]])/maxevts
        

    print
    print "%.5f :  TOTAL TIME"%sum
    print "\n"

    # Print out info for individual subtasks

    print "Time taken by each subtask"
    print "--------------------------------------------"
    mysubkeys=mySubTime.keys()
    mysubkeys.sort()
    mysubkeys.reverse()
    subsum=0
    for i in mysubkeys:
        print "%.5f : \t%s\t%i events"%(i,mySubTime[i],len(myTimers[mySubTime[i]]))
        subsum+=i*len(myTimers[mySubTime[i]])/maxevts
    print "%.5f : TOTAL SUBPROCESS TIME"%subsum
    print "\n\n"

