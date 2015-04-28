#!/usr/bin/env python

#----------------------------------------------
def printHelp():
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


#----------------------------------------------
def getTime(line):
    time = line.split(" ")[1]
    time = time.split(":")
    time = int(time[0])*60*60+int(time[1])*60+float(time[2])
    return time

#Stream states
kStarted=0
kFinished=1

#----------------------------------------------
def readLogFile(fileName):
  f = open(fileName,"r")

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
              processingSteps.append(("FINISH INIT",kFinished,stream,getTime(l)-startTime,False))
      if l.find("processing event for module") != -1:
          time = getTime(l)
          if startTime == 0:
              startTime = time
          time = time - startTime
          trans = kStarted
          stream = 0
          delayed = False
          if l.find("finished:") != -1:
              trans = kFinished
          stream = int( l[l.find("stream = ")+9])
          name = l.split("'")[1]
          if l.find("delayed") != -1:
              delayed = True
          if len(name) > maxNameSize:
              maxNameSize = len(name)
          processingSteps.append((name,trans,stream,time,delayed))
          if stream > numStreams:
              numStreams = stream
  f.close()
  return (processingSteps,numStreams,maxNameSize)


#----------------------------------------------
def findStalledModules(processingSteps, numStreams):
  streamTime = [0]*(numStreams+1)
  stalledModules = {}
  for n,trans,s,time,delayed in processingSteps:
    waitTime = None
    if delayed:
      n = "source"
    if trans == kStarted:
      waitTime = time - streamTime[s]
    else:
      streamTime[s] = time
    if waitTime is not None:
      if waitTime > 0.1:
        t = stalledModules.setdefault(n,[])
        t.append(waitTime)
  return stalledModules


#----------------------------------------------
def createAsciiImage(processingSteps, numStreams, maxNameSize):
  #print processingSteps
  #exit(1)
  streamState = [1]*(numStreams+1)
  streamTime = [0]*(numStreams+1)
  #lastTime = 0
  seenInit = False
  for n,trans,s,time,delayed in processingSteps:
      if n == "FINISH INIT":
          seenInit = True
          continue
      oldState = streamState[s]
      streamState[s]=trans
      waitTime = None
      if delayed:
          n = "source"
      if trans == kStarted:
          waitTime = time - streamTime[s]
          streamState[s]=2
      else:
          if oldState != trans:
              streamState[s]=3
          streamTime[s] = time
      states = "%-*s: " % (maxNameSize,n)
      for state in streamState:
          if state == 0:
              states +=" "
          elif state == 1:
              states +="|"
          elif state == 2:
              states +="-"
          elif state == 3:
              states +="+"
      if waitTime is not None:
          states += " %.2f"% waitTime
          if waitTime > 0.1 and seenInit:
              states += " STALLED"

      print states
      streamState[s]=trans
  return stalledModules

#----------------------------------------------
def printStalledModulesInOrder(stalledModules):
  priorities = list()
  for n,t in stalledModules.iteritems():
    t.sort(reverse=True)
    priorities.append((n,sum(t),t))

  def sumSort(i,j):
    return cmp(i[1],j[1])
  priorities.sort(cmp=sumSort, reverse=True)

  for n,s,t in priorities:
    print n, "%.2f"%s, [ "%.2f"%x for x in t]


def createPDFImage(processingSteps, numStreams, stalledModuleInfo):
  import matplotlib.pyplot as plt
  
  streamStartDepth = [0]*(numStreams+1)
  streamTime = [0]*(numStreams+1)
  streamStartTimes = [ [] for x in xrange(numStreams+1)]
  streamColors = [[] for x in xrange(numStreams+1)]

  stalledModuleNames = [ x for x in stalledModuleInfo.iterkeys()]
  
  fig, ax = plt.subplots()
  
  streamStartTimes = [ [] for x in xrange(numStreams+1)]
  streamColors = [[] for x in xrange(numStreams+1)]
  
  for n,trans,s,time,delayed in processingSteps:
    if trans == kStarted:
      streamStartDepth[s] +=1
      streamTime[s] = time
    else:
      streamStartDepth[s] -=1
      if 0 == streamStartDepth[s]:
        streamStartTimes[s].append((streamTime[s],time-streamTime[s]))
        c="green"
        if delayed:
          c="orange"
        elif n in stalledModuleNames:
          c="red"
        #elif len(streamColors[s]) %2:
        #  c="blue"
        streamColors[s].append(c)
  i=1
  for s in xrange(numStreams+1):
    t = streamStartTimes[s]
    ax.broken_barh(t,(i-0.4,0.8),facecolors=streamColors[s],edgecolors=streamColors[s],linewidth=0)
    i=i+1
  plt.savefig("stall.pdf")



#=======================================
if __name__=="__main__":
  import sys

  if len(sys.argv) == 1:
    printHelp()
    exit(0)
  
  doGraphic = False
  if len(sys.argv) == 3:
    if sys.argv[1] == '-g':
      doGraphic = True
    else:
      print "unknown argument ",sys.argv[1]
      exit(-1)
  fileName =sys.argv[-1]

  processingSteps,numStreams,maxNameSize = readLogFile(sys.argv[-1])
  stalledModules = findStalledModules(processingSteps, numStreams)
  if not doGraphic:
    createAsciiImage(processingSteps, numStreams, maxNameSize)
  else:
    createPDFImage(processingSteps, numStreams, stalledModules)
  printStalledModulesInOrder(stalledModules)

