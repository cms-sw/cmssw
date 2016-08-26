#!/usr/bin/env python
import sys

#----------------------------------------------
def printHelp():
    s = """Purpose: Convert a cmsRun log with Tracer info into a stream stall graph.

edmStreamStallGrapher [-g] <log file name>

Option: -g instead of ascii art, create a pdf file showing the work being done on each stream

To Use: Add the Tracer Service to the cmsRun job you want to check for stream stalls.
 Make sure to use the 'printTimstamps' option
    cms.Service("Tracer", printTimestamps = cms.untracked.bool(True))
 After running the job, execute this script and pass the name of the log file to the
 script as the only command line argument.
 
To Read: The script will then print an 'ASCII art' stall graph which consists of the name of
 the module which either started or stopped running on a stream, and the number of modules
 running on each stream at that the moment in time. If the module just started, you will also
 see the amount of time the module spent between finishing its prefetching and starting.
 The state of a module is represented by a symbol:
   plus  ("+") the stream has just finished waiting and is starting a module
   minus ("-") the stream just finished running a module
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
kPrefetchEnd=2

#Special names
kSourceFindEvent = "sourceFindEvent"
kSourceDelayedRead ="sourceDelayedRead"
kFinishInit = "FINISH INIT"

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
        processingSteps.append((kFinishInit,kFinished,stream,getTime(l)-startTime))
    if l.find("processing event :") != -1:
      time = getTime(l)
      if startTime == 0:
        startTime = time
      time = time - startTime
      streamIndex = l.find("stream = ")
      stream = int( l[streamIndex+9:l.find(" ",streamIndex+10)])
      name = kSourceFindEvent
      trans = kFinished
      #the start of an event is the end of the framework part
      if l.find("starting:") != -1:
        trans = kStarted
      processingSteps.append((name,trans,stream,time))
      if stream > numStreams:
        numStreams = stream
    if l.find("processing event for module") != -1:
      time = getTime(l)
      if startTime == 0:
        startTime = time
      time = time - startTime
      trans = kStarted
      stream = 0
      delayed = False
      if l.find("finished:") != -1:
        if l.find("prefetching") != -1:
          trans = kPrefetchEnd
        else:
            trans = kFinished
      else:
        if l.find("prefetching") != -1:
          #skip this since we don't care about prefetch starts
          continue
      streamIndex = l.find("stream = ")
      stream = int( l[streamIndex+9:l.find(" ",streamIndex+10)])
      name = l.split("'")[1]
      if len(name) > maxNameSize:
        maxNameSize = len(name)
      processingSteps.append((name,trans,stream,time))
      if stream > numStreams:
        numStreams = stream
    if l.find("event delayed read from source") != -1:
      time = getTime(l)
      if startTime == 0:
        startTime = time
      time = time - startTime
      trans = kStarted
      stream = 0
      delayed = False
      if l.find("finished:") != -1:
        trans = kFinished
      streamIndex = l.find("stream = ")
      stream = int( l[streamIndex+9:l.find(" ",streamIndex+10)])
      name = kSourceDelayedRead
      if len(name) > maxNameSize:
        maxNameSize = len(name)
      processingSteps.append((name,trans,stream,time))
      if stream > numStreams:
        numStreams = stream

  f.close()
  return (processingSteps,numStreams,maxNameSize)


#----------------------------------------------
# Patterns:
#
# source: The source just records how long it was spent doing work, not how long it was stalled. Can get a lower
#   bound on the stall time by measuring the time the stream was doing no work up till the source was run.
# modules: time between prefetch finished and 'start processing' is the time it took to acquire any resources
#
#
def findStalledModules(processingSteps, numStreams):
  streamTime = [0]*(numStreams+1)
  stalledModules = {}
  previousStartWasADelayed = [False]*(numStreams+1)
  modulesActiveOnStream = [ {} for x in xrange(0,numStreams+1)]
  for n,trans,s,time in processingSteps:
    waitTime = None
    modulesOnStream =modulesActiveOnStream[s]
    if trans == kPrefetchEnd:
      modulesOnStream[n]=time
    if trans == kStarted:
      if n in modulesOnStream:
        waitTime = time - modulesOnStream[n]
      if n == kSourceDelayedRead:
        if 0 == len(modulesOnStream):
          waitTime = time - streamTime[s]
    if trans == kFinished:
      if n != kSourceDelayedRead and n!=kSourceFindEvent and n!=kFinishInit:
        del modulesOnStream[n]
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
  streamTime = [0]*(numStreams+1)
  streamState = [0]*(numStreams+1)
  modulesActiveOnStreams = [{} for x in xrange(0,numStreams+1)]
  #lastTime = 0
  seenInit = False
  for n,trans,s,time in processingSteps:
      if n == kFinishInit:
          seenInit = True
          continue
      modulesActiveOnStream = modulesActiveOnStreams[s]
      waitTime = None
      if trans == kPrefetchEnd:
        modulesActiveOnStream[n] = time
        continue
      if trans == kStarted:
        if n != kSourceFindEvent:
          streamState[s] +=1
        if n in modulesActiveOnStream:
          waitTime = time - modulesActiveOnStream[n]
        if n == kSourceDelayedRead:
          if streamState[s] == 0:
            waitTime = time-streamTime[s]
      if trans == kFinished:
        if n != kSourceDelayedRead and n!=kSourceFindEvent:
          del modulesActiveOnStream[n]
        if n != kSourceFindEvent:
          streamState[s] -=1
        streamTime[s] = time
      states = "%-*s: " % (maxNameSize,n)
      if trans == kStarted:
        states +="+ "
      if trans == kFinished:
        states +="- "
      for index, state in enumerate(streamState):
        if n==kSourceFindEvent and index == s:
          states +="* "
        else:
          states +=str(state)+" "
      if waitTime is not None:
          states += " %.2f"% waitTime
          if waitTime > 0.1 and seenInit:
              states += " STALLED "+str(time)+" "+str(s)

      print states
  return stalledModules

#----------------------------------------------
def printStalledModulesInOrder(stalledModules):
  priorities = list()
  maxNameSize = 0
  for n,t in stalledModules.iteritems():
    nameLength = len(n)
    if nameLength > maxNameSize:
      maxNameSize = nameLength
    t.sort(reverse=True)
    priorities.append((n,sum(t),t))

  def sumSort(i,j):
    return cmp(i[1],j[1])
  priorities.sort(cmp=sumSort, reverse=True)

  nameColumn = "Stalled Module"
  if len(nameColumn) > maxNameSize:
    maxNameSize = len(nameColumn)
  
  stallColumn = "Tot Stall Time"
  stallColumnLength = len(stallColumn)
  
  print "%-*s" % (maxNameSize, nameColumn), "%-*s"%(stallColumnLength,stallColumn), " Stall Times"
  for n,s,t in priorities:
    paddedName = "%-*s:" % (maxNameSize,n)
    print paddedName, "%-*.2f"%(stallColumnLength,s), ", ".join([ "%.2f"%x for x in t])


def createPDFImage(processingSteps, numStreams, stalledModuleInfo):
  import matplotlib.pyplot as plt
  
  stalledModuleNames = set([ x for x in stalledModuleInfo.iterkeys()])
  
  streamStartTimes = [ [] for x in xrange(numStreams+1)]
  streamColors = [[] for x in xrange(numStreams+1)]
  modulesActiveOnStreams = [{} for x in xrange(0,numStreams+1)]
  streamLastEventEndTimes = [None]*(numStreams+1)
  streamMultipleModulesRunnningTimes = [[] for x in xrange(numStreams+1)]
  maxNumberOfConcurrentModulesOnAStream = 0
  streamInvertedMessageFromModule = [ set() for x in xrange(numStreams+1)]

  for n,trans,s,time in processingSteps:
    startTime = None
    if streamLastEventEndTimes[s] is None:
      streamLastEventEndTimes[s]=time
    if n == kFinishInit:
      continue
    if trans == kStarted:
      if n == kSourceFindEvent:
        #we assume the time from the end of the last event for a
        # stream until the start of a new event for that stream
        # is taken up by the source
        startTime = streamLastEventEndTimes[s]
        moduleNames = set(n)
      else:
        activeModules =modulesActiveOnStreams[s]
        moduleNames = set(activeModules.iterkeys())
        if n in streamInvertedMessageFromModule[s]:
            #this is the rare case where a finished message is issued
            # before the corresponding started
            streamInvertedMessageFromModule[s].remove(n)
            continue
        activeModules[n]=time
        nModulesRunning = len(activeModules)
        if nModulesRunning > 1:
          streamMultipleModulesRunnningTimes[s].append([nModulesRunning, time, None])
          if nModulesRunning > maxNumberOfConcurrentModulesOnAStream:
              maxNumberOfConcurrentModulesOnAStream = nModulesRunning
          #need to create a new time span to avoid overlaps in graph
          startTime = min( activeModules.itervalues() )
          #print s, startTime, time, moduleNames
          for k in activeModules.iterkeys():
            activeModules[k]=time
            
    if trans == kFinished:
      if n == kSourceFindEvent:
        streamLastEventEndTimes[s]=time
      else:
        activeModules =modulesActiveOnStreams[s]
        if n not in activeModules:
            #this is the rare case where a finished message is issued
            # before the corresponding started
            streamInvertedMessageFromModule[s].add(n)
            continue
        startTime = activeModules[n]
        moduleNames = set(activeModules.iterkeys())
        del activeModules[n]
        nModulesRunning = len(activeModules)
        if nModulesRunning > 0:
          streamMultipleModulesRunnningTimes[s][-1][2]=time
          #reset start time for remaining modules to this time
          # to avoid overlapping time ranges when making the plot
          for k in activeModules.iterkeys():
            activeModules[k] = time
    if startTime is not None:
      c="green"
      if (kSourceDelayedRead in moduleNames) or (kSourceFindEvent in moduleNames):
          c = "orange"
      streamStartTimes[s].append((startTime,time-startTime))
      for n in moduleNames:
        if n in stalledModuleNames:
          c="red"
          break
        #elif len(streamColors[s]) %2:
        #  c="blue"
      streamColors[s].append(c)
  
  #consolodate contiguous blocks with the same color
  # this drastically reduces the size of the pdf file
  oldStreamTimes = streamStartTimes
  oldStreamColors = streamColors

  streamStartTimes = [ [] for x in xrange(numStreams+1)]
  streamColors = [[] for x in xrange(numStreams+1)]
  
  for s in xrange(numStreams+1):
    streamStartTimes[s].append(oldStreamTimes[s][0])
    streamColors[s].append(oldStreamColors[s][0])
    lastStartTime,lastTimeLength = oldStreamTimes[s][0]
    lastColor = oldStreamColors[s][0]
    for i in xrange(1, len(oldStreamTimes[s])):
      start,length = oldStreamTimes[s][i]
      color = oldStreamColors[s][i]
      #use a millisecond tolerance to avoid rounding
      if color == lastColor and abs(lastStartTime+lastTimeLength-start)<0.001:
        lastTimeLength += length
      else:
        streamStartTimes[s].append((lastStartTime,lastTimeLength))
        streamColors[s].append(lastColor)
        lastStartTime = start
        lastTimeLength = length
        lastColor = color
    streamStartTimes[s].append((lastStartTime,lastTimeLength))
    streamColors[s].append(lastColor)

  fig, ax = plt.subplots()
  ax.set_xlabel("Time (sec)")
  ax.set_ylabel("Stream ID")

  i=1
  for s in xrange(numStreams+1):
    t = streamStartTimes[s]
    ax.broken_barh(t,(i-0.4,0.8),facecolors=streamColors[s],edgecolors=streamColors[s],linewidth=0)
    i=i+1
#now superimpose the number of concurrently running modules on to the graph
  if maxNumberOfConcurrentModulesOnAStream > 1:
    height = 0.8/maxNumberOfConcurrentModulesOnAStream
    i=1
    for s in xrange(numStreams+1):
      occurences = streamMultipleModulesRunnningTimes[s]
      for info in occurences:
        if info[2] is None:
          continue
        times = (info[1], info[2]-info[1])
        ax.broken_barh( [times],(i-0.4, height*(info[0]-1)), facecolors="blue",edgecolors="blue",linewidth=0)
      i+=1
  #add key .1, .3, .7
  fig.text(0.1, 0.95, "modules running", color = "green", horizontalalignment = 'left')
  fig.text(0.5, 0.95, "stalled module running", color = "red", horizontalalignment = 'center')
  fig.text(0.9, 0.95, "read from input", color = "orange", horizontalalignment = 'right')
  fig.text(0.5, 0.92, "multiple modules running", color = "blue", horizontalalignment = 'center')
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

  sys.stderr.write( ">reading file\n" )
  processingSteps,numStreams,maxNameSize = readLogFile(sys.argv[-1])
  sys.stderr.write(">processing data\n")
  stalledModules = findStalledModules(processingSteps, numStreams)
  if not doGraphic:
    sys.stderr.write(">preparing ASCII art\n")
    createAsciiImage(processingSteps, numStreams, maxNameSize)
  else:
    sys.stderr.write(">creating PDF\n")
    createPDFImage(processingSteps, numStreams, stalledModules)
  printStalledModulesInOrder(stalledModules)

