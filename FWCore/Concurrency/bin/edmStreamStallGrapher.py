#!/usr/bin/env python

import sys

if len(sys.argv) == 1:
    s = """Purpose: Convert a cmsRun log with Tracer info into a stream stall graph.

To Use: Add the Tracer Service to the cmsRun job you want to check for stream stalls.
 Make sure to use the 'printTimstamps' option
    cms.Service("Tracer", printTimestamps = cms.untracked.bool(True))
 After running the job, execute this script and pass the name of the log file to the
 script as the only command line argument.
 
To Read: The script will then print an 'ASCII art' stall graph which consists of the name of
 the module which either started or stopped running on a stream, and the state of each
 stream at that moment in time and if the module just started, you will also see the
 amount of time on that stream between the previous module finishing and this module starting.
 The state of a stream is represented by a symbol:
   blank (" ") the stream is currently running a module
   line  ("|") the stream is waiting to run a module
   star  ("*") the stream has just finished waiting and is starting a module
 If a module had to wait more than 0.1 seconds, the end of the line will have "STALLED".
 Once the first 4 events have finished processing, the program prints "FINISH INIT".
 This is useful if one wants to ignore stalled caused by startup actions, e.g. reading
 conditions.

 Once the graph is completed, the program outputs the list of modules which had
 the greatest total stall times. The list is sorted by total stall time and 
 written in descending order. In addition, the list of all stall times for the
 module is given.
"""
    print s
    exit(0)

fileName = sys.argv[1]

f = open(fileName,"r")

def getTime(line):
    time = line.split(" ")[1]
    time = time.split(":")
    time = int(time[0])*60*60+int(time[1])*60+float(time[2])
    return time

processingSteps = list()
numStreams = 0
maxNameSize = 0
startTime = 0
foundEventToStartFrom = False
for l in f:
    if not foundEventToStartFrom:
        if l.find("event = 5") != -1:
            foundEventToStartFrom = True
            stream = int( l[l.find("stream = ")+9])
            processingSteps.append(("FINISH INIT",1,stream,getTime(l)-startTime))
    if l.find("processing event for module") != -1:
        time = getTime(l)
        if startTime == 0:
            startTime = time
        time = time - startTime
        trans = 0
        stream = 0
        if l.find("finished:") != -1:
            trans = 1
        stream = int( l[l.find("stream = ")+9])
        name = l.split("'")[1]
        if len(name) > maxNameSize:
            maxNameSize = len(name)
        processingSteps.append((name,trans,stream,time))
        if stream > numStreams:
            numStreams = stream
f.close()


#print processingSteps
#exit(1)
streamState = [1]*(numStreams+1)
streamTime = [0]*(numStreams+1)
#lastTime = 0
seenInit = False
stalledModules = {}
for n,trans,s,time in processingSteps:
    if n == "FINISH INIT":
        seenInit = True
    streamState[s]=trans
    waitTime = None
    if not trans:
        waitTime = time - streamTime[s]
        streamState[s]=2
    else:
        streamTime[s] = time
    states = "%-*s: " % (maxNameSize,n)
    for state in streamState:
        if state == 0:
            states +=" "
        elif state == 1:
            states +="|"
        elif state == 2:
            states +="*"
    if waitTime is not None:
        states += " %.2f"% waitTime
        if waitTime > 0.1 and seenInit:
            t = stalledModules.setdefault(n,[])
            t.append(waitTime)
            states += " STALLED"

    print states
    streamState[s]=trans

priorities = list()
for n,t in stalledModules.iteritems():
    t.sort(reverse=True)
    priorities.append((n,sum(t),t))

def sumSort(i,j):
    return cmp(i[1],j[1])
priorities.sort(cmp=sumSort, reverse=True)

for n,s,t in priorities:
    print n, "%.2f"%s, [ "%.2f"%x for x in t]
