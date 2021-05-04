#!/usr/bin/env python
from __future__ import print_function
from builtins import range
from itertools import groupby
from operator import attrgetter,itemgetter
import sys
from collections import defaultdict
import six
#----------------------------------------------
def printHelp():
    s = '''
To Use: Add the StallMonitor Service to the cmsRun job you want to check for
  stream stalls. Use something like this in the configuration:

  process.add_(cms.Service("StallMonitor", fileName = cms.untracked.string("stallMonitor.log")))

  After running the job, execute this script and pass the name of the
  StallMonitor log file to the script.

  By default, the script will then print an 'ASCII art' stall graph
  which consists of a line of text for each time a module or the
  source stops or starts. Each line contains the name of the module
  which either started or stopped running, and the number of modules
  running on each stream at that moment in time. After that will be
  the time and stream number. Then if a module just started, you
  will also see the amount of time the module spent between finishing
  its prefetching and starting.  The state of a module is represented
  by a symbol:

    plus  ("+") the stream has just finished waiting and is starting a module
    minus ("-") the stream just finished running a module

  If a module had to wait more than 0.1 seconds, the end of the line
  will have "STALLED". Startup actions, e.g. reading conditions,
  may affect results for the first few events.

  Using the command line arguments described above you can make the
  program create a PDF file with actual graphs instead of the 'ASCII art'
  output.

  Once the graph is completed, the program outputs the list of modules
  which had the greatest total stall times. The list is sorted by
  total stall time and written in descending order. In addition, the
  list of all stall times for the module is given.

  There is an inferior alternative (an old obsolete way).
  Instead of using the StallMonitor Service, you can use the
  Tracer Service.  Make sure to use the 'printTimestamps' option
  cms.Service("Tracer", printTimestamps = cms.untracked.bool(True))
  There are problems associated with this and it is not recommended.'''
    return s

kStallThreshold=100000 #in microseconds
kTracerInput=False

#Stream states
kStarted=0
kFinished=1
kPrefetchEnd=2
kStartedAcquire=3
kFinishedAcquire=4
kStartedSource=5
kFinishedSource=6
kStartedSourceDelayedRead=7
kFinishedSourceDelayedRead=8

#Special names
kSourceFindEvent = "sourceFindEvent"
kSourceDelayedRead ="sourceDelayedRead"

#----------------------------------------------
def processingStepsFromStallMonitorOutput(f,moduleNames):
    for rawl in f:
        l = rawl.strip()
        if not l or l[0] == '#':
            continue
        (step,payload) = tuple(l.split(None,1))
        payload=payload.split()

        # Ignore these
        if step == 'E' or step == 'e':
            continue

        # Payload format is:
        #  <stream id> <..other fields..> <time since begin job>
        stream = int(payload[0])
        time = int(payload[-1])
        trans = None
        isEvent = True

        name = None
        # 'S' = begin of event creation in source
        # 's' = end of event creation in source
        if step == 'S' or step == 's':
            name = kSourceFindEvent
            trans = kStartedSource
            # The start of an event is the end of the framework part
            if step == 's':
                trans = kFinishedSource
        else:
            # moduleID is the second payload argument for all steps below
            moduleID = payload[1]

            # 'p' = end of module prefetching
            # 'M' = begin of module processing
            # 'm' = end of module processing
            if step == 'p' or step == 'M' or step == 'm':
                trans = kStarted
                if step == 'p':
                    trans = kPrefetchEnd
                elif step == 'm':
                    trans = kFinished
                if step == 'm' or step == 'M':
                    isEvent = (int(payload[2]) == 0)
                name = moduleNames[moduleID]

            # 'A' = begin of module acquire function
            # 'a' = end of module acquire function
            elif step == 'A' or step == 'a':
                trans = kStartedAcquire
                if step == 'a':
                    trans = kFinishedAcquire
                name = moduleNames[moduleID]

            # Delayed read from source
            # 'R' = begin of delayed read from source
            # 'r' = end of delayed read from source
            elif step == 'R' or step == 'r':
                trans = kStartedSourceDelayedRead
                if step == 'r':
                    trans = kFinishedSourceDelayedRead
                name = kSourceDelayedRead

        if trans is not None:
            yield (name,trans,stream,time, isEvent)
    
    return

class StallMonitorParser(object):
    def __init__(self,f):
        numStreams = 0
        numStreamsFromSource = 0
        moduleNames = {}
        for rawl in f:
            l = rawl.strip()
            if l and l[0] == 'M':
                i = l.split(' ')
                if i[3] == '4':
                    #found global begin run
                    numStreams = int(i[1])+1
                    break
            if numStreams == 0 and l and l[0] == 'S':
                s = int(l.split(' ')[1])
                if s > numStreamsFromSource:
                  numStreamsFromSource = s
            if len(l) > 5 and l[0:2] == "#M":
                (id,name)=tuple(l[2:].split())
                moduleNames[id] = name
                continue
        self._f = f
        if numStreams == 0:
          numStreams = numStreamsFromSource +1
        self.numStreams =numStreams
        self._moduleNames = moduleNames
        self.maxNameSize =0
        for n in six.iteritems(moduleNames):
            self.maxNameSize = max(self.maxNameSize,len(n))
        self.maxNameSize = max(self.maxNameSize,len(kSourceDelayedRead))

    def processingSteps(self):
        """Create a generator which can step through the file and return each processing step.
        Using a generator reduces the memory overhead when parsing a large file.
            """
        self._f.seek(0)
        return processingStepsFromStallMonitorOutput(self._f,self._moduleNames)

#----------------------------------------------
# Utility to get time out of Tracer output text format
def getTime(line):
    time = line.split(" ")[1]
    time = time.split(":")
    time = int(time[0])*60*60+int(time[1])*60+float(time[2])
    time = int(1000000*time) # convert to microseconds
    return time

#----------------------------------------------
# The next function parses the Tracer output.
# Here are some differences to consider if you use Tracer output
# instead of the StallMonitor output.
# - The time in the text of the Tracer output is not as precise
# as the StallMonitor (.01 s vs .001 s)
# - The MessageLogger bases the time on when the message printed
# and not when it was initially queued up to print which smears
# the accuracy of the times.
# - Both of the previous things can produce some strange effects
# in the output plots.
# - The file size of the Tracer text file is much larger.
# - The CPU work needed to parse the Tracer files is larger.
# - The Tracer log file is expected to have "++" in the first
# or fifth line. If there are extraneous lines at the beginning
# you have to remove them.
# - The ascii printout out will have one extraneous line
# near the end for the SourceFindEvent start.
# - The only advantage I can see is that you have only
# one output file to handle instead of two, the regular
# log file and the StallMonitor output.
# We might should just delete the Tracer option because it is
# clearly inferior ...
def parseTracerOutput(f):
    processingSteps = []
    numStreams = 0
    maxNameSize = 0
    startTime = 0
    streamsThatSawFirstEvent = set()
    for l in f:
        trans = None
        # We estimate the start and stop of the source
        # by the end of the previous event and start of
        # the event. This is historical, probably because
        # the Tracer output for the begin and end of the
        # source event does not include the stream number.
        if l.find("processing event :") != -1:
            name = kSourceFindEvent
            trans = kStartedSource
            # the end of the source is estimated using the start of the event
            if l.find("starting:") != -1:
                trans = kFinishedSource
        elif l.find("processing event for module") != -1:
            trans = kStarted
            if l.find("finished:") != -1:
                if l.find("prefetching") != -1:
                    trans = kPrefetchEnd
                else:
                    trans = kFinished
            else:
                if l.find("prefetching") != -1:
                    #skip this since we don't care about prefetch starts
                    continue
            name = l.split("'")[1]
        elif l.find("processing event acquire for module:") != -1:
            trans = kStartedAcquire
            if l.find("finished:") != -1:
                trans = kFinishedAcquire
            name = l.split("'")[1]
        elif l.find("event delayed read from source") != -1:
            trans = kStartedSourceDelayedRead
            if l.find("finished:") != -1:
                trans = kFinishedSourceDelayedRead
            name = kSourceDelayedRead
        if trans is not None:
            time = getTime(l)
            if startTime == 0:
                startTime = time
            time = time - startTime
            streamIndex = l.find("stream = ")
            stream = int(l[streamIndex+9:l.find(" ",streamIndex+10)])
            maxNameSize = max(maxNameSize, len(name))

            if trans == kFinishedSource and not stream in streamsThatSawFirstEvent:
                # This is wrong but there is no way to estimate the time better
                # because there is no previous event for the first event.
                processingSteps.append((name,kStartedSource,stream,time,True))
                streamsThatSawFirstEvent.add(stream)

            processingSteps.append((name,trans,stream,time, True))
            numStreams = max(numStreams, stream+1)

    f.close()
    return (processingSteps,numStreams,maxNameSize)

class TracerParser(object):
    def __init__(self,f):
        self._processingSteps,self.numStreams,self.maxNameSize = parseTracerOutput(f)
    def processingSteps(self):
        return self._processingSteps

#----------------------------------------------
def chooseParser(inputFile):

    firstLine = inputFile.readline().rstrip()
    for i in range(3):
        inputFile.readline()
    # Often the Tracer log file starts with 4 lines not from the Tracer
    fifthLine = inputFile.readline().rstrip()
    inputFile.seek(0) # Rewind back to beginning
    if (firstLine.find("# Transition") != -1) or (firstLine.find("# Step") != -1):
        print("> ... Parsing StallMonitor output.")
        return StallMonitorParser

    if firstLine.find("++") != -1 or fifthLine.find("++") != -1:
        global kTracerInput
        kTracerInput = True
        print("> ... Parsing Tracer output.")
        return TracerParser
    else:
        inputFile.close()
        print("Unknown input format.")
        exit(1)

#----------------------------------------------
def readLogFile(inputFile):
    parseInput = chooseParser(inputFile)
    return parseInput(inputFile)

#----------------------------------------------
#
# modules: The time between prefetch finished and 'start processing' is
#   the time it took to acquire any resources which is by definition the
#   stall time.
#
# source: The source just records how long it spent doing work,
#   not how long it was stalled. We can get a lower bound on the stall
#   time for delayed reads by measuring the time the stream was doing
#   no work up till the start of the source delayed read.
#
def findStalledModules(processingSteps, numStreams):
    streamTime = [0]*numStreams
    streamState = [0]*numStreams
    stalledModules = {}
    modulesActiveOnStream = [{} for x in range(numStreams)]
    for n,trans,s,time,isEvent in processingSteps:

        waitTime = None
        modulesOnStream = modulesActiveOnStream[s]
        if trans == kPrefetchEnd:
            modulesOnStream[n] = time
        elif trans == kStarted or trans == kStartedAcquire:
            if n in modulesOnStream:
                waitTime = time - modulesOnStream[n]
                modulesOnStream.pop(n, None)
            streamState[s] +=1
        elif trans == kFinished or trans == kFinishedAcquire:
            streamState[s] -=1
            streamTime[s] = time
        elif trans == kStartedSourceDelayedRead:
            if streamState[s] == 0:
                waitTime = time - streamTime[s]
        elif trans == kStartedSource:
            modulesOnStream.clear()
        elif trans == kFinishedSource or trans == kFinishedSourceDelayedRead:
            streamTime[s] = time
        if waitTime is not None:
            if waitTime > kStallThreshold:
                t = stalledModules.setdefault(n,[])
                t.append(waitTime)
    return stalledModules


def createModuleTiming(processingSteps, numStreams):
    import json 
    streamTime = [0]*numStreams
    streamState = [0]*numStreams
    moduleTimings = defaultdict(list)
    modulesActiveOnStream = [defaultdict(int) for x in range(numStreams)]
    for n,trans,s,time,isEvent in processingSteps:
        waitTime = None
        modulesOnStream = modulesActiveOnStream[s]
        if isEvent:
            if trans == kStarted:
                streamState[s] = 1
                modulesOnStream[n]=time
            elif trans == kFinished:
                waitTime = time - modulesOnStream[n]
                modulesOnStream.pop(n, None)
                streamState[s] = 0
                moduleTimings[n].append(float(waitTime/1000.))

    with open('module-timings.json', 'w') as outfile:
        outfile.write(json.dumps(moduleTimings, indent=4))

#----------------------------------------------
def createAsciiImage(processingSteps, numStreams, maxNameSize):
    streamTime = [0]*numStreams
    streamState = [0]*numStreams
    modulesActiveOnStreams = [{} for x in range(numStreams)]
    for n,trans,s,time,isEvent in processingSteps:
        waitTime = None
        modulesActiveOnStream = modulesActiveOnStreams[s]
        if trans == kPrefetchEnd:
            modulesActiveOnStream[n] = time
            continue
        elif trans == kStartedAcquire or trans == kStarted:
            if n in modulesActiveOnStream:
                waitTime = time - modulesActiveOnStream[n]
                modulesActiveOnStream.pop(n, None)
            streamState[s] +=1
        elif trans == kFinishedAcquire or trans == kFinished:
            streamState[s] -=1
            streamTime[s] = time
        elif trans == kStartedSourceDelayedRead:
            if streamState[s] == 0:
                waitTime = time - streamTime[s]
        elif trans == kStartedSource:
            modulesActiveOnStream.clear()
        elif trans == kFinishedSource or trans == kFinishedSourceDelayedRead:
            streamTime[s] = time
        states = "%-*s: " % (maxNameSize,n)
        if trans == kStartedAcquire or trans == kStarted or trans == kStartedSourceDelayedRead or trans == kStartedSource:
            states +="+ "
        else:
            states +="- "
        for index, state in enumerate(streamState):
            if n==kSourceFindEvent and index == s:
                states +="* "
            else:
                states +=str(state)+" "
        states += " -- " + str(time/1000.) + " " + str(s) + " "
        if waitTime is not None:
            states += " %.2f"% (waitTime/1000.)
            if waitTime > kStallThreshold:
                states += " STALLED"

        print(states)

#----------------------------------------------
def printStalledModulesInOrder(stalledModules):
    priorities = []
    maxNameSize = 0
    for name,t in six.iteritems(stalledModules):
        maxNameSize = max(maxNameSize, len(name))
        t.sort(reverse=True)
        priorities.append((name,sum(t),t))

    priorities.sort(key=lambda a: a[1], reverse=True)

    nameColumn = "Stalled Module"
    maxNameSize = max(maxNameSize, len(nameColumn))

    stallColumn = "Tot Stall Time"
    stallColumnLength = len(stallColumn)

    print("%-*s" % (maxNameSize, nameColumn), "%-*s"%(stallColumnLength,stallColumn), " Stall Times")
    for n,s,t in priorities:
        paddedName = "%-*s:" % (maxNameSize,n)
        print(paddedName, "%-*.2f"%(stallColumnLength,s/1000.), ", ".join([ "%.2f"%(x/1000.) for x in t]))

#--------------------------------------------------------
class Point:
    def __init__(self, x_, y_):
        self.x = x_
        self.y = y_

    def __str__(self):
        return "(x: {}, y: {})".format(self.x,self.y)

    def __repr__(self):
        return self.__str__()

#--------------------------------------------------------
def reduceSortedPoints(ps):
    if len(ps) < 2:
        return ps
    reducedPoints = []
    tmp = Point(ps[0].x, ps[0].y)
    for p in ps[1:]:
        if tmp.x == p.x:
            tmp.y += p.y
        else:
            reducedPoints.append(tmp)
            tmp = Point(p.x, p.y)
    reducedPoints.append(tmp)
    reducedPoints = [p for p in reducedPoints if p.y != 0]
    return reducedPoints

# -------------------------------------------
def adjacentDiff(*pairLists):
    points = []
    for pairList in pairLists:
        points += [Point(x[0], 1) for x in pairList if x[1] != 0]
        points += [Point(sum(x),-1) for x in pairList if x[1] != 0]
    points.sort(key=attrgetter('x'))
    return points

stackType = 'stack'

# --------------------------------------------
class Stack:
    def __init__(self):
        self.data = []

    def update(self, graphType, points):
        tmp = points
        if len(self.data) != 0:
            tmp += self.data[-1][1]

        tmp.sort(key=attrgetter('x'))
        tmp = reduceSortedPoints(tmp)
        self.data.append((graphType, tmp))

#---------------------------------------------
# StreamInfoElement
class StreamInfoElement:
    def __init__(self, begin_, delta_, color_):
        self.begin=begin_
        self.delta=delta_
        self.color=color_

    def unpack(self):
        return self.begin, self.delta, self.color

#----------------------------------------------
# Consolidating contiguous blocks with the same color
# drastically reduces the size of the pdf file.
def consolidateContiguousBlocks(numStreams, streamInfo):
    oldStreamInfo = streamInfo
    streamInfo = [[] for x in range(numStreams)]

    for s in range(numStreams):
        if oldStreamInfo[s]:
            lastStartTime,lastTimeLength,lastColor = oldStreamInfo[s][0].unpack()
            for info in oldStreamInfo[s][1:]:
                start,length,color = info.unpack()
                if color == lastColor and lastStartTime+lastTimeLength == start:
                    lastTimeLength += length
                else:
                    streamInfo[s].append(StreamInfoElement(lastStartTime,lastTimeLength,lastColor))
                    lastStartTime = start
                    lastTimeLength = length
                    lastColor = color
            streamInfo[s].append(StreamInfoElement(lastStartTime,lastTimeLength,lastColor))

    return streamInfo

#----------------------------------------------
# Consolidating contiguous blocks with the same color drastically
# reduces the size of the pdf file.  Same functionality as the
# previous function, but with slightly different implementation.
def mergeContiguousBlocks(blocks):
    oldBlocks = blocks

    blocks = []
    if not oldBlocks:
        return blocks

    lastStartTime,lastTimeLength,lastHeight = oldBlocks[0]
    for start,length,height in oldBlocks[1:]:
        if height == lastHeight and lastStartTime+lastTimeLength == start:
            lastTimeLength += length
        else:
            blocks.append((lastStartTime,lastTimeLength,lastHeight))
            lastStartTime = start
            lastTimeLength = length
            lastHeight = height
    blocks.append((lastStartTime,lastTimeLength,lastHeight))

    return blocks

#----------------------------------------------
def plotPerStreamAboveFirstAndPrepareStack(points, allStackTimes, ax, stream, height, streamHeightCut, doPlot, addToStackTimes, color, threadOffset):
    points = sorted(points, key=attrgetter('x'))
    points = reduceSortedPoints(points)
    streamHeight = 0
    preparedTimes = []
    for t1,t2 in zip(points, points[1:]):
        streamHeight += t1.y
        # We make a cut here when plotting because the first row for
        # each stream was already plotted previously and we do not
        # need to plot it again. And also we want to count things
        # properly in allStackTimes. We want to avoid double counting
        # or missing running modules and this is complicated because
        # we counted the modules in the first row already.
        if streamHeight < streamHeightCut:
            continue
        preparedTimes.append((t1.x,t2.x-t1.x, streamHeight))
    preparedTimes.sort(key=itemgetter(2))
    preparedTimes = mergeContiguousBlocks(preparedTimes)

    for nthreads, ts in groupby(preparedTimes, itemgetter(2)):
        theTS = [(t[0],t[1]) for t in ts]
        if doPlot:
            theTimes = [(t[0]/1000000.,t[1]/1000000.) for t in theTS]
            yspan = (stream-0.4+height,height*(nthreads-1))
            ax.broken_barh(theTimes, yspan, facecolors=color, edgecolors=color, linewidth=0)
        if addToStackTimes:
            allStackTimes[color].extend(theTS*(nthreads-threadOffset))

#----------------------------------------------
def createPDFImage(pdfFile, shownStacks, processingSteps, numStreams, stalledModuleInfo, displayExternalWork, checkOrder, setXAxis, xLower, xUpper):

    stalledModuleNames = set([x for x in stalledModuleInfo.iterkeys()])
    streamLowestRow = [[] for x in range(numStreams)]
    modulesActiveOnStreams = [set() for x in range(numStreams)]
    acquireActiveOnStreams = [set() for x in range(numStreams)]
    externalWorkOnStreams  = [set() for x in range(numStreams)]
    previousFinishTime = [None for x in range(numStreams)]
    streamRunningTimes = [[] for x in range(numStreams)]
    streamExternalWorkRunningTimes = [[] for x in range(numStreams)]
    maxNumberOfConcurrentModulesOnAStream = 1
    externalWorkModulesInJob = False
    previousTime = [0 for x in range(numStreams)]

    # The next five variables are only used to check for out of order transitions
    finishBeforeStart = [set() for x in range(numStreams)]
    finishAcquireBeforeStart = [set() for x in range(numStreams)]
    countSource = [0 for x in range(numStreams)]
    countDelayedSource = [0 for x in range(numStreams)]
    countExternalWork = [defaultdict(int) for x in range(numStreams)]

    timeOffset = None
    for n,trans,s,time,isEvent in processingSteps:
        if timeOffset is None:
            timeOffset = time
        startTime = None
        time -=timeOffset
        # force the time to monotonically increase on each stream
        if time < previousTime[s]:
            time = previousTime[s]
        previousTime[s] = time

        activeModules = modulesActiveOnStreams[s]
        acquireModules = acquireActiveOnStreams[s]
        externalWorkModules = externalWorkOnStreams[s]

        if trans == kStarted or trans == kStartedSourceDelayedRead or trans == kStartedAcquire or trans == kStartedSource :
            if checkOrder:
                # Note that the code which checks the order of transitions assumes that
                # all the transitions exist in the input. It is checking only for order
                # problems, usually a start before a finish. Problems are fixed and
                # silently ignored. Nothing gets plotted for transitions that are
                # in the wrong order.
                if trans == kStarted:
                    countExternalWork[s][n] -= 1
                    if n in finishBeforeStart[s]:
                        finishBeforeStart[s].remove(n)
                        continue
                elif trans == kStartedAcquire:
                    if n in finishAcquireBeforeStart[s]:
                        finishAcquireBeforeStart[s].remove(n)
                        continue

            if trans == kStartedSourceDelayedRead:
                countDelayedSource[s] += 1
                if countDelayedSource[s] < 1:
                    continue
            elif trans == kStartedSource:
                countSource[s] += 1
                if countSource[s] < 1:
                    continue

            moduleNames = activeModules.copy()
            moduleNames.update(acquireModules)
            if trans == kStartedAcquire:
                 acquireModules.add(n)
            else:
                 activeModules.add(n)
            streamRunningTimes[s].append(Point(time,1))
            if moduleNames or externalWorkModules:
                startTime = previousFinishTime[s]
            previousFinishTime[s] = time

            if trans == kStarted and n in externalWorkModules:
                externalWorkModules.remove(n)
                streamExternalWorkRunningTimes[s].append(Point(time, -1))
            else:
                nTotalModules = len(activeModules) + len(acquireModules) + len(externalWorkModules)
                maxNumberOfConcurrentModulesOnAStream = max(maxNumberOfConcurrentModulesOnAStream, nTotalModules)
        elif trans == kFinished or trans == kFinishedSourceDelayedRead or trans == kFinishedAcquire or trans == kFinishedSource :
            if checkOrder:
                if trans == kFinished:
                    if n not in activeModules:
                        finishBeforeStart[s].add(n)
                        continue

            if trans == kFinishedSourceDelayedRead:
                countDelayedSource[s] -= 1
                if countDelayedSource[s] < 0:
                    continue
            elif trans == kFinishedSource:
                countSource[s] -= 1
                if countSource[s] < 0:
                    continue

            if trans == kFinishedAcquire:
                if checkOrder:
                    countExternalWork[s][n] += 1
                if displayExternalWork:
                    externalWorkModulesInJob = True
                    if (not checkOrder) or countExternalWork[s][n] > 0:
                        externalWorkModules.add(n)
                        streamExternalWorkRunningTimes[s].append(Point(time,+1))
                if checkOrder and n not in acquireModules:
                    finishAcquireBeforeStart[s].add(n)
                    continue
            streamRunningTimes[s].append(Point(time,-1))
            startTime = previousFinishTime[s]
            previousFinishTime[s] = time
            moduleNames = activeModules.copy()
            moduleNames.update(acquireModules)

            if trans == kFinishedAcquire:
                acquireModules.remove(n)
            elif trans == kFinishedSourceDelayedRead:
                if countDelayedSource[s] == 0:
                    activeModules.remove(n)
            elif trans == kFinishedSource:
                if countSource[s] == 0:
                    activeModules.remove(n)
            else:
                activeModules.remove(n)

        if startTime is not None:
            c="green"
            if not isEvent:
              c="limegreen"
            if not moduleNames:
                c = "darkviolet"
            elif (kSourceDelayedRead in moduleNames) or (kSourceFindEvent in moduleNames):
                c = "orange"
            else:
                for n in moduleNames:
                    if n in stalledModuleNames:
                        c="red"
                        break
            streamLowestRow[s].append(StreamInfoElement(startTime, time-startTime, c))
    streamLowestRow = consolidateContiguousBlocks(numStreams, streamLowestRow)

    nr = 1
    if shownStacks:
        nr += 1
    fig, ax = plt.subplots(nrows=nr, squeeze=True)
    axStack = None
    if shownStacks:
        [xH,yH] = fig.get_size_inches()
        fig.set_size_inches(xH,yH*4/3)
        ax = plt.subplot2grid((4,1),(0,0), rowspan=3)
        axStack = plt.subplot2grid((4,1),(3,0))

    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Stream ID")
    ax.set_ylim(-0.5,numStreams-0.5)
    ax.yaxis.set_ticks(range(numStreams))
    if (setXAxis):
        ax.set_xlim((xLower, xUpper))

    height = 0.8/maxNumberOfConcurrentModulesOnAStream
    allStackTimes={'green': [],'limegreen':[], 'red': [], 'blue': [], 'orange': [], 'darkviolet': []}
    for iStream,lowestRow in enumerate(streamLowestRow):
        times=[(x.begin/1000000., x.delta/1000000.) for x in lowestRow] # Scale from microsec to sec.
        colors=[x.color for x in lowestRow]
        # for each stream, plot the lowest row
        ax.broken_barh(times,(iStream-0.4,height),facecolors=colors,edgecolors=colors,linewidth=0)
        # record them also for inclusion in the stack plot
        # the darkviolet ones get counted later so do not count them here
        for info in lowestRow:
            if not info.color == 'darkviolet':
                allStackTimes[info.color].append((info.begin, info.delta))

    # Now superimpose the number of concurrently running modules on to the graph.
    if maxNumberOfConcurrentModulesOnAStream > 1 or externalWorkModulesInJob:

        for i,perStreamRunningTimes in enumerate(streamRunningTimes):

            perStreamTimesWithExtendedWork = list(perStreamRunningTimes)
            perStreamTimesWithExtendedWork.extend(streamExternalWorkRunningTimes[i])

            plotPerStreamAboveFirstAndPrepareStack(perStreamTimesWithExtendedWork,
                                                   allStackTimes, ax, i, height,
                                                   streamHeightCut=2,
                                                   doPlot=True,
                                                   addToStackTimes=False,
                                                   color='darkviolet',
                                                   threadOffset=1)

            plotPerStreamAboveFirstAndPrepareStack(perStreamRunningTimes,
                                                   allStackTimes, ax, i, height,
                                                   streamHeightCut=2,
                                                   doPlot=True,
                                                   addToStackTimes=True,
                                                   color='blue',
                                                   threadOffset=1)

            plotPerStreamAboveFirstAndPrepareStack(streamExternalWorkRunningTimes[i],
                                                   allStackTimes, ax, i, height,
                                                   streamHeightCut=1,
                                                   doPlot=False,
                                                   addToStackTimes=True,
                                                   color='darkviolet',
                                                   threadOffset=0)

    if shownStacks:
        print("> ... Generating stack")
        stack = Stack()
        for color in ['green','limegreen','blue','red','orange','darkviolet']:
            tmp = allStackTimes[color]
            tmp = reduceSortedPoints(adjacentDiff(tmp))
            stack.update(color, tmp)

        for stk in reversed(stack.data):
            color = stk[0]

            # Now arrange list in a manner that it can be grouped by the height of the block
            height = 0
            xs = []
            for p1,p2 in zip(stk[1], stk[1][1:]):
                height += p1.y
                xs.append((p1.x, p2.x-p1.x, height))
            xs.sort(key = itemgetter(2))
            xs = mergeContiguousBlocks(xs)

            for height, xpairs in groupby(xs, itemgetter(2)):
                finalxs = [(e[0]/1000000.,e[1]/1000000.) for e in xpairs]
                # plot the stacked plot, one color and one height on each call to broken_barh
                axStack.broken_barh(finalxs, (0, height), facecolors=color, edgecolors=color, linewidth=0)

        axStack.set_xlabel("Time (sec)");
        axStack.set_ylabel("# modules");
        axStack.set_xlim(ax.get_xlim())
        axStack.tick_params(top='off')

    fig.text(0.1, 0.95, "modules running event", color = "green", horizontalalignment = 'left')
    fig.text(0.1, 0.92, "modules running other", color = "limegreen", horizontalalignment = 'left')
    fig.text(0.5, 0.95, "stalled module running", color = "red", horizontalalignment = 'center')
    fig.text(0.9, 0.95, "read from input", color = "orange", horizontalalignment = 'right')
    fig.text(0.5, 0.92, "multiple modules running", color = "blue", horizontalalignment = 'center')
    if displayExternalWork:
        fig.text(0.9, 0.92, "external work", color = "darkviolet", horizontalalignment = 'right')
    print("> ... Saving to file: '{}'".format(pdfFile))
    plt.savefig(pdfFile)

#=======================================
if __name__=="__main__":
    import argparse
    import re
    import sys

    # Program options
    parser = argparse.ArgumentParser(description='Convert a text file created by cmsRun into a stream stall graph.',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=printHelp())
    parser.add_argument('filename',
                        type=argparse.FileType('r'), # open file
                        help='file to process')
    parser.add_argument('-g', '--graph',
                        nargs='?',
                        metavar="'stall.pdf'",
                        const='stall.pdf',
                        dest='graph',
                        help='''Create pdf file of stream stall graph.  If -g is specified
                        by itself, the default file name is \'stall.pdf\'.  Otherwise, the
                        argument to the -g option is the filename.''')
    parser.add_argument('-s', '--stack',
                        action='store_true',
                        help='''Create stack plot, combining all stream-specific info.
                        Can be used only when -g is specified.''')
    parser.add_argument('-e', '--external',
                        action='store_false',
                        help='''Suppress display of external work in graphs.''')
    parser.add_argument('-o', '--order',
                        action='store_true',
                        help='''Enable checks for and repair of transitions in the input that are in the wrong order (for example a finish transition before a corresponding start). This is always enabled for Tracer input, but is usually an unnecessary waste of CPU time and memory with StallMonitor input and by default not enabled.''')
    parser.add_argument('-t', '--timings',
                        action='store_true',
                        help='''Create a dictionary of module labels and their timings from the stall monitor log. Write the dictionary filea as a json file modules-timings.json.''')
    parser.add_argument('-l', '--lowerxaxis',
                        type=float,
                        default=0.0,
                        help='''Lower limit of x axis, default 0, not used if upper limit not set''')
    parser.add_argument('-u', '--upperxaxis',
                        type=float,
                        help='''Upper limit of x axis, if not set then x axis limits are set automatically''')
    args = parser.parse_args()

    # Process parsed options
    inputFile = args.filename
    pdfFile = args.graph
    shownStacks = args.stack
    displayExternalWork = args.external
    checkOrder = args.order
    doModuleTimings = False
    if args.timings:
        doModuleTimings = True

    setXAxis = False
    xUpper = 0.0
    if args.upperxaxis is not None:
        setXAxis = True
        xUpper = args.upperxaxis
    xLower = args.lowerxaxis

    doGraphic = False
    if pdfFile is not None:
        doGraphic = True
        import matplotlib
        # Need to force display since problems with CMSSW matplotlib.
        matplotlib.use("PDF")
        import matplotlib.pyplot as plt
        if not re.match(r'^[\w\.]+$', pdfFile):
            print("Malformed file name '{}' supplied with the '-g' option.".format(pdfFile))
            print("Only characters 0-9, a-z, A-Z, '_', and '.' are allowed.")
            exit(1)

        if '.' in pdfFile:
            extension = pdfFile.split('.')[-1]
            supported_filetypes = plt.figure().canvas.get_supported_filetypes()
            if not extension in supported_filetypes:
                print("A graph cannot be saved to a filename with extension '{}'.".format(extension))
                print("The allowed extensions are:")
                for filetype in supported_filetypes:
                    print("   '.{}'".format(filetype))
                exit(1)

    if pdfFile is None and shownStacks:
        print("The -s (--stack) option can be used only when the -g (--graph) option is specified.")
        exit(1)

    sys.stderr.write(">reading file: '{}'\n".format(inputFile.name))
    reader = readLogFile(inputFile)
    if kTracerInput:
        checkOrder = True
    sys.stderr.write(">processing data\n")
    stalledModules = findStalledModules(reader.processingSteps(), reader.numStreams)


    if not doGraphic:
        sys.stderr.write(">preparing ASCII art\n")
        createAsciiImage(reader.processingSteps(), reader.numStreams, reader.maxNameSize)
    else:
        sys.stderr.write(">creating PDF\n")
        createPDFImage(pdfFile, shownStacks, reader.processingSteps(), reader.numStreams, stalledModules, displayExternalWork, checkOrder, setXAxis, xLower, xUpper)
    printStalledModulesInOrder(stalledModules)
    if doModuleTimings:
        sys.stderr.write(">creating module-timings.json\n")
        createModuleTiming(reader.processingSteps(), reader.numStreams)
