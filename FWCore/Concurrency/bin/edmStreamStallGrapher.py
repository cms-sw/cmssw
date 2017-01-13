#!/usr/bin/env python
from itertools import groupby
from operator import attrgetter,itemgetter
import sys

#----------------------------------------------
def printHelp():
    s = '''
To Use: Add the Tracer Service to the cmsRun job you want to check for
  stream stalls.  Make sure to use the 'printTimstamps' option
    cms.Service("Tracer", printTimestamps = cms.untracked.bool(True))
  After running the job, execute this script and pass the name of the
  log file to the script as the only command line argument.

To Read: The script will then print an 'ASCII art' stall graph which
  consists of the name of the module which either started or stopped
  running on a stream, and the number of modules running on each
  stream at that the moment in time. If the module just started, you
  will also see the amount of time the module spent between finishing
  its prefetching and starting.  The state of a module is represented
  by a symbol:

    plus  ("+") the stream has just finished waiting and is starting a module
    minus ("-") the stream just finished running a module

  If a module had to wait more than 0.1 seconds, the end of the line
  will have "STALLED".  Once the first 4 events have finished
  processing, the program prints "FINISH INIT".  This is useful if one
  wants to ignore stalled caused by startup actions, e.g. reading
  conditions.

  Once the graph is completed, the program outputs the list of modules
  which had the greatest total stall times. The list is sorted by
  total stall time and written in descending order. In addition, the
  list of all stall times for the module is given.'''
    return s


kStallThreshold=100 #in milliseconds
kTracerInput=False

#Stream states
kStarted=0
kFinished=1
kPrefetchEnd=2

#Special names
kSourceFindEvent = "sourceFindEvent"
kSourceDelayedRead ="sourceDelayedRead"

#----------------------------------------------
def parseStallMonitorOutput(f):
    processingSteps = []
    numStreams = 0
    maxNameSize = 0
    foundEventToStartFrom = False
    moduleNames = {}
    for rawl in f:
        l = rawl.strip()
        if not l or l[0] == '#':
            if len(l) > 5 and l[0:2] == "#M":
                (id,name)=tuple(l[2:].split())
                moduleNames[id] = name
            continue
        (step,payload) = tuple(l.split(None,1))
        payload=payload.split()

        # Payload format is:
        #  <stream id> <..other fields..> <time since begin job>
        stream = int(payload[0])
        time = int(payload[-1])
        trans = None

        numStreams = max(numStreams, stream+1)

        # 'S' = begin of event creation in source
        # 's' = end of event creation in source
        if step == 'S' or step == 's':
            name = kSourceFindEvent
            trans = kStarted
            # The start of an event is the end of the framework part
            if step == 's':
                trans = kFinished
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
                name = moduleNames[moduleID]

            # Delayed read from source
            # 'R' = begin of delayed read from source
            # 'r' = end of delayed read from source
            if step == 'R' or step == 'r':
                trans = kStarted
                if step == 'r':
                    trans = kFinished
                name = kSourceDelayedRead

            maxNameSize = max(maxNameSize, len(name))
            processingSteps.append((name,trans,stream,time))

    f.close()
    return (processingSteps,numStreams,maxNameSize)

#----------------------------------------------
def getTime(line):
    time = line.split(" ")[1]
    time = time.split(":")
    time = int(time[0])*60*60+int(time[1])*60+float(time[2])
    time = int(1000*time) # convert to milliseconds
    return time

#----------------------------------------------
def parseTracerOutput(f):
    processingSteps = []
    numStreams = 0
    maxNameSize = 0
    startTime = 0
    for l in f:
        if l.find("processing event :") != -1:
            time = getTime(l)
            if startTime == 0:
                startTime = time
            time = time - startTime
            streamIndex = l.find("stream = ")
            stream = int(l[streamIndex+9:l.find(" ",streamIndex+10)])
            name = kSourceFindEvent
            trans = kFinished
            #the start of an event is the end of the framework part
            if l.find("starting:") != -1:
                trans = kStarted
            processingSteps.append((name,trans,stream,time))
            numStreams = max(numStreams, stream+1)
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
            maxNameSize = max(maxNameSize, len(name))
            processingSteps.append((name,trans,stream,time))
            numStreams = max(numStreams, stream+1)
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
            stream = int(l[streamIndex+9:l.find(" ",streamIndex+10)])
            name = kSourceDelayedRead
            maxNameSize = max(maxNameSize, len(name))
            processingSteps.append((name,trans,stream,time))
            numStreams = max(numStreams, stream+1)
    f.close()
    return (processingSteps,numStreams,maxNameSize)


#----------------------------------------------
def chooseParser(inputFile):
    firstLine = inputFile.readline().rstrip()
    inputFile.seek(0) # Rewind back to beginning

    if firstLine.find("# Step") != -1:
        print "> ... Parsing StallMonitor output."
        return parseStallMonitorOutput
    elif firstLine.find("++") != -1:
        global kTracerInput
        kTracerInput = True
        print "> ... Parsing Tracer output."
        return parseTracerOutput
    else:
        inputFile.close()
        print "Unknown input format."
        exit(1)

#----------------------------------------------
def readLogFile(inputFile):
    parseInput = chooseParser(inputFile)
    return parseInput(inputFile)

#----------------------------------------------
# Patterns:
#
# source: The source just records how long it was spent doing work,
#   not how long it was stalled. We can get a lower bound on the stall
#   time by measuring the time the stream was doing no work up till
#   the source was run.
# modules: The time between prefetch finished and 'start processing' is
#   the time it took to acquire any resources
#
def findStalledModules(processingSteps, numStreams):
    streamTime = [0]*numStreams
    stalledModules = {}
    modulesActiveOnStream = [{} for x in xrange(numStreams)]
    for n,trans,s,time in processingSteps:
        waitTime = None
        modulesOnStream = modulesActiveOnStream[s]
        if trans == kPrefetchEnd:
            modulesOnStream[n] = time
        if trans == kStarted:
            if n in modulesOnStream:
                waitTime = time - modulesOnStream[n]
            if n == kSourceDelayedRead:
                if 0 == len(modulesOnStream):
                    waitTime = time - streamTime[s]
        if trans == kFinished:
            if n != kSourceDelayedRead and n!=kSourceFindEvent:
                del modulesOnStream[n]
            streamTime[s] = time
        if waitTime is not None:
            if waitTime > kStallThreshold:
                t = stalledModules.setdefault(n,[])
                t.append(waitTime)
    return stalledModules


#----------------------------------------------
def createAsciiImage(processingSteps, numStreams, maxNameSize):
    streamTime = [0]*numStreams
    streamState = [0]*numStreams
    modulesActiveOnStreams = [{} for x in xrange(numStreams)]
    for n,trans,s,time in processingSteps:
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
            states += " %.2f"% (waitTime/1000.)
            if waitTime > kStallThreshold:
                states += " STALLED "+str(time/1000.)+" "+str(s)

        print states
    return stalledModules

#----------------------------------------------
def printStalledModulesInOrder(stalledModules):
    priorities = []
    maxNameSize = 0
    for name,t in stalledModules.iteritems():
        maxNameSize = max(maxNameSize, len(name))
        t.sort(reverse=True)
        priorities.append((name,sum(t),t))

    def sumSort(i,j):
        return cmp(i[1],j[1])
    priorities.sort(cmp=sumSort, reverse=True)

    nameColumn = "Stalled Module"
    maxNameSize = max(maxNameSize, len(nameColumn))

    stallColumn = "Tot Stall Time"
    stallColumnLength = len(stallColumn)

    print "%-*s" % (maxNameSize, nameColumn), "%-*s"%(stallColumnLength,stallColumn), " Stall Times"
    for n,s,t in priorities:
        paddedName = "%-*s:" % (maxNameSize,n)
        print paddedName, "%-*.2f"%(stallColumnLength,s/1000.), ", ".join([ "%.2f"%(x/1000.) for x in t])

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
    tmp = ps[0]
    for p in ps[1:]:
        if tmp.x == p.x:
            tmp.y += p.y
        else:
            reducedPoints.append(tmp)
            tmp = p
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
    streamInfo = [[] for x in xrange(numStreams)]

    for s in xrange(numStreams):
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
def createPDFImage(pdfFile, shownStacks, processingSteps, numStreams, stalledModuleInfo):

    stalledModuleNames = set([x for x in stalledModuleInfo.iterkeys()])
    streamInfo = [[] for x in xrange(numStreams)]
    modulesActiveOnStreams = [{} for x in xrange(numStreams)]
    streamLastEventEndTimes = [None]*numStreams
    streamRunningTimes = [[] for x in xrange(numStreams)]
    maxNumberOfConcurrentModulesOnAStream = 1
    streamInvertedMessageFromModule = [set() for x in xrange(numStreams)]

    for n,trans,s,time in processingSteps:
        startTime = None
        if streamLastEventEndTimes[s] is None:
            streamLastEventEndTimes[s]=time
        if trans == kStarted:
            if n == kSourceFindEvent:
                # We assume the time from the end of the last event
                # for a stream until the start of a new event for that
                # stream is taken up by the source.
                startTime = streamLastEventEndTimes[s]
                moduleNames = set(n)
            else:
                activeModules = modulesActiveOnStreams[s]
                moduleNames = set(activeModules.iterkeys())
                if n in streamInvertedMessageFromModule[s] and kTracerInput:
                    # This is the rare case where a finished message
                    # is issued before the corresponding started.
                    streamInvertedMessageFromModule[s].remove(n)
                    continue
                activeModules[n] = time
                nModulesRunning = len(activeModules)
                streamRunningTimes[s].append(Point(time,1))
                maxNumberOfConcurrentModulesOnAStream = max(maxNumberOfConcurrentModulesOnAStream, nModulesRunning)
                if nModulesRunning > 1:
                    # Need to create a new time span to avoid overlaps in graph.
                    startTime = min(activeModules.itervalues())
                    for k in activeModules.iterkeys():
                        activeModules[k]=time

        if trans == kFinished:
            if n == kSourceFindEvent:
                streamLastEventEndTimes[s]=time
            else:
                activeModules = modulesActiveOnStreams[s]
                if n not in activeModules and kTracerInput:
                    # This is the rare case where a finished message
                    # is issued before the corresponding started.
                    streamInvertedMessageFromModule[s].add(n)
                    continue
                streamRunningTimes[s].append(Point(time,-1))
                startTime = activeModules[n]
                moduleNames = set(activeModules.iterkeys())
                del activeModules[n]
                nModulesRunning = len(activeModules)
                if nModulesRunning > 0:
                    # Reset start time for remaining modules to this time
                    # to avoid overlapping time ranges when making the plot.
                    for k in activeModules.iterkeys():
                        activeModules[k] = time
        if startTime is not None:
            c="green"
            if (kSourceDelayedRead in moduleNames) or (kSourceFindEvent in moduleNames):
                c = "orange"
            for n in moduleNames:
                if n in stalledModuleNames:
                    c="red"
                    break
            streamInfo[s].append(StreamInfoElement(startTime, time-startTime, c))

    streamInfo = consolidateContiguousBlocks(numStreams, streamInfo)

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
    ax.yaxis.set_ticks(xrange(numStreams))

    height = 0.8/maxNumberOfConcurrentModulesOnAStream
    allStackTimes={'green': [], 'red': [], 'blue': [], 'orange': []}
    for i,perStreamInfo in enumerate(streamInfo):
        times=[(x.begin/1000., x.delta/1000.) for x in perStreamInfo] # Scale from msec to sec.
        colors=[x.color for x in perStreamInfo]
        ax.broken_barh(times,(i-0.4,height),facecolors=colors,edgecolors=colors,linewidth=0)
        for info in perStreamInfo:
            allStackTimes[info.color].append((info.begin, info.delta))

    # Now superimpose the number of concurrently running modules on to the graph.
    if maxNumberOfConcurrentModulesOnAStream > 1:

        for i,perStreamRunningTimes in enumerate(streamRunningTimes):
            perStreamTimes = sorted(perStreamRunningTimes, key=attrgetter('x'))
            perStreamTimes = reduceSortedPoints(perStreamTimes)
            streamHeight = 0
            preparedTimes = []
            for t1,t2 in zip(perStreamTimes, perStreamTimes[1:]):
                streamHeight += t1.y
                if streamHeight < 2:
                    continue
                preparedTimes.append((t1.x,t2.x-t1.x, streamHeight))
            preparedTimes.sort(key=itemgetter(2))
            preparedTimes = mergeContiguousBlocks(preparedTimes)
            for nthreads, ts in groupby(preparedTimes, itemgetter(2)):
                theTS = [(t[0],t[1]) for t in ts]
                theTimes = [(t[0]/1000.,t[1]/1000.) for t in theTS]
                yspan = (i-0.4+height,height*(nthreads-1))
                ax.broken_barh(theTimes, yspan, facecolors='blue', edgecolors='blue', linewidth=0)
                allStackTimes['blue'].extend(theTS*(nthreads-1))

    if shownStacks:
        print "> ... Generating stack"
        stack = Stack()
        for color in ['green','blue','red','orange']:
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
                finalxs = [(e[0]/1000.,e[1]/1000.) for e in xpairs]
                axStack.broken_barh(finalxs, (0, height), facecolors=color, edgecolors=color, linewidth=0)

        axStack.set_xlabel("Time (sec)");
        axStack.set_ylabel("# threads");
        axStack.set_xlim(ax.get_xlim())
        axStack.tick_params(top='off')

    fig.text(0.1, 0.95, "modules running", color = "green", horizontalalignment = 'left')
    fig.text(0.5, 0.95, "stalled module running", color = "red", horizontalalignment = 'center')
    fig.text(0.9, 0.95, "read from input", color = "orange", horizontalalignment = 'right')
    fig.text(0.5, 0.92, "multiple modules running", color = "blue", horizontalalignment = 'center')
    print "> ... Saving to file: '{}'".format(pdfFile)
    plt.savefig(pdfFile)

#=======================================
if __name__=="__main__":
    import argparse
    import re
    import sys

    # Program options
    parser = argparse.ArgumentParser(description='Convert a cmsRun log with Tracer info into a stream stall graph.',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=printHelp())
    parser.add_argument('filename',
                        type=argparse.FileType('r'), # open file
                        help='log file to process')
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
    args = parser.parse_args()

    # Process parsed options
    inputFile = args.filename
    pdfFile = args.graph
    shownStacks = args.stack

    doGraphic = False
    if pdfFile is not None:
        doGraphic = True
        import matplotlib
        # Need to force display since problems with CMSSW matplotlib.
        matplotlib.use("PDF")
        import matplotlib.pyplot as plt
        if not re.match(r'^[\w\.]+$', pdfFile):
            print "Malformed file name '{}' supplied with the '-g' option.".format(pdfFile)
            print "Only characters 0-9, a-z, A-Z, '_', and '.' are allowed."
            exit(1)

        if '.' in pdfFile:
            extension = pdfFile.split('.')[-1]
            supported_filetypes = plt.figure().canvas.get_supported_filetypes()
            if not extension in supported_filetypes:
                print "A graph cannot be saved to a filename with extension '{}'.".format(extension)
                print "The allowed extensions are:"
                for filetype in supported_filetypes:
                    print "   '.{}'".format(filetype)
                exit(1)

    if pdfFile is None and shownStacks:
        print "The -s (--stack) option can be used only when the -g (--graph) option is specified."
        exit(1)

    sys.stderr.write(">reading file: '{}'\n".format(inputFile.name))
    processingSteps,numStreams,maxNameSize = readLogFile(inputFile)
    sys.stderr.write(">processing data\n")
    stalledModules = findStalledModules(processingSteps, numStreams)
    if not doGraphic:
        sys.stderr.write(">preparing ASCII art\n")
        createAsciiImage(processingSteps, numStreams, maxNameSize)
    else:
        sys.stderr.write(">creating PDF\n")
        createPDFImage(pdfFile, shownStacks, processingSteps, numStreams, stalledModules)
    printStalledModulesInOrder(stalledModules)
