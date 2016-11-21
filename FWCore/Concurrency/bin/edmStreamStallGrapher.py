#!/usr/bin/env python
import re
import sys

#----------------------------------------------
def printHelp():
    s = """Purpose: Convert a cmsRun log with Tracer info into a stream stall graph.

edmStreamStallGrapher [-g[=arg]] <log file name>

Options: -g[=arg] instead of ascii art, create a pdf file of name
         'arg' showing the work being done on each stream.  If '=arg'
         is not specified, the pdf file name is 'stall.pdf'.  There
         can be no spaces before and after the '=' sign.

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
  list of all stall times for the module is given.  """
    print s


kStallThreshold=100 #in milliseconds
kTracerInput=False

#Stream states
kStarted=0
kFinished=1
kPrefetchEnd=2

#Special names
kSourceFindEvent = "sourceFindEvent"
kSourceDelayedRead ="sourceDelayedRead"
kFinishInit = "FINISH INIT"

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

        if stream > numStreams:
            numStreams = stream

        if not foundEventToStartFrom:
            # Event number is second from the end for the 'E' step
            if step == 'E' and payload[-2] == '5':
                foundEventToStartFrom = True
                processingSteps.append((kFinishInit,kFinished,stream,time))
                continue

        # 'E' = begin of event processing
        # 'e' = end of event processing
        if step == 'E' or step == 'e':
            name = kSourceFindEvent
            trans = kStarted
            # The start of an event is the end of the framework part
            if step == 'e':
                trans = kFinished

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

        if len(name) > maxNameSize:
            maxNameSize = len(name)

        processingSteps.append((name,trans,stream,time))

    f.close()
    return (processingSteps,numStreams,maxNameSize)

#----------------------------------------------
def getTime(line):
    time = line.split(" ")[1]
    time = time.split(":")
    time = int(time[0])*60*60+int(time[1])*60+float(time[2])
    time = 1000*time # convert to milliseconds
    return time

#----------------------------------------------
def parseTracerOutput(f):
    processingSteps = []
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
def readLogFile(fileName):
    f = open(fileName,"r")
    parseInput = chooseParser(f)
    return parseInput(f)

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
    streamTime = [0]*(numStreams+1)
    stalledModules = {}
    modulesActiveOnStream = [{} for x in xrange(0,numStreams+1)]
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
            if n != kSourceDelayedRead and n!=kSourceFindEvent and n!=kFinishInit:
                del modulesOnStream[n]
            streamTime[s] = time
        if waitTime is not None:
            if waitTime > kStallThreshold:
                t = stalledModules.setdefault(n,[])
                t.append(waitTime)
    return stalledModules


#----------------------------------------------
def createAsciiImage(processingSteps, numStreams, maxNameSize):
    streamTime = [0]*(numStreams+1)
    streamState = [0]*(numStreams+1)
    modulesActiveOnStreams = [{} for x in xrange(0,numStreams+1)]
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
            states += " %.2f"% (waitTime/1000.)
            if waitTime > kStallThreshold and seenInit:
                states += " STALLED "+str(time/1000.)+" "+str(s)

        print states
    return stalledModules

#----------------------------------------------
def printStalledModulesInOrder(stalledModules):
    priorities = []
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
        print paddedName, "%-*.2f"%(stallColumnLength,s/1000.), ", ".join([ "%.2f"%(x/1000.) for x in t])

#----------------------------------------------
# Consolidating contiguous blocks with the same color
# drastically reduces the size of the pdf file.
def consolidateContiguousBlocks(numStreams, streamTimes, streamColors):
    oldStreamTimes = streamTimes
    oldStreamColors = streamColors

    streamTimes = [[] for x in xrange(numStreams+1)]
    streamColors = [[] for x in xrange(numStreams+1)]

    for s in xrange(numStreams+1):
        lastStartTime,lastTimeLength = oldStreamTimes[s][0]
        lastColor = oldStreamColors[s][0]
        for i in xrange(1, len(oldStreamTimes[s])):
            start,length = oldStreamTimes[s][i]
            color = oldStreamColors[s][i]
            if color == lastColor and lastStartTime+lastTimeLength == start:
                lastTimeLength += length
            else:
                streamTimes[s].append((lastStartTime,lastTimeLength))
                streamColors[s].append(lastColor)
                lastStartTime = start
                lastTimeLength = length
                lastColor = color
        streamTimes[s].append((lastStartTime,lastTimeLength))
        streamColors[s].append(lastColor)

    return (streamTimes,streamColors)

#----------------------------------------------
def createPDFImage(pdfFile, processingSteps, numStreams, stalledModuleInfo):
    # Need to force display since problems with CMSSW matplotlib.
    import matplotlib
    matplotlib.use("PDF")
    import matplotlib.pyplot as plt

    stalledModuleNames = set([ x for x in stalledModuleInfo.iterkeys()])

    streamTimes = [[] for x in xrange(numStreams+1)]
    streamColors = [[] for x in xrange(numStreams+1)]
    modulesActiveOnStreams = [{} for x in xrange(0,numStreams+1)]
    streamLastEventEndTimes = [None]*(numStreams+1)
    streamMultipleModulesRunningTimes = [[] for x in xrange(numStreams+1)]
    maxNumberOfConcurrentModulesOnAStream = 0
    streamInvertedMessageFromModule = [set() for x in xrange(numStreams+1)]

    for n,trans,s,time in processingSteps:
        if n == kFinishInit:
            continue
        startTime = None
        if streamLastEventEndTimes[s] is None:
            streamLastEventEndTimes[s]=time
        if n == kFinishInit:
            continue
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
                if nModulesRunning > 1:
                    streamMultipleModulesRunningTimes[s].append([nModulesRunning, time, None])
                    if nModulesRunning > maxNumberOfConcurrentModulesOnAStream:
                        maxNumberOfConcurrentModulesOnAStream = nModulesRunning
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
                startTime = activeModules[n]
                moduleNames = set(activeModules.iterkeys())
                del activeModules[n]
                nModulesRunning = len(activeModules)
                if nModulesRunning > 0:
                    streamMultipleModulesRunningTimes[s][-1][2]=time
                    # Reset start time for remaining modules to this time
                    # to avoid overlapping time ranges when making the plot.
                    for k in activeModules.iterkeys():
                        activeModules[k] = time
        if startTime is not None:
            c="green"
            if (kSourceDelayedRead in moduleNames) or (kSourceFindEvent in moduleNames):
                c = "orange"
            streamTimes[s].append((startTime,time-startTime))
            for n in moduleNames:
                if n in stalledModuleNames:
                    c="red"
                    break
            streamColors[s].append(c)


    (streamTimes,streamColors) = consolidateContiguousBlocks(numStreams, streamTimes, streamColors)

    fig, ax = plt.subplots()
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Stream ID")

    height = 0.8/maxNumberOfConcurrentModulesOnAStream

    def scaleToSeconds(times):
        return [(x[0]/1000.,x[1]/1000.) for x in times]

    for i,times in enumerate(streamTimes):
        ax.broken_barh(scaleToSeconds(times),(i-0.4,height),facecolors=streamColors[i],edgecolors=streamColors[i],linewidth=0)

    # Now superimpose the number of concurrently running modules on to the graph.
    if maxNumberOfConcurrentModulesOnAStream > 1:
        for i,occurrences in enumerate(streamMultipleModulesRunningTimes):
            for info in occurrences:
                if info[2] is None:
                    continue
                times = (info[1], info[2]-info[1])
                ax.broken_barh(scaleToSeconds([times]), (i-0.4+height, height*(info[0]-1)), facecolors="blue",edgecolors="blue",linewidth=0)

    fig.text(0.1, 0.95, "modules running", color = "green", horizontalalignment = 'left')
    fig.text(0.5, 0.95, "stalled module running", color = "red", horizontalalignment = 'center')
    fig.text(0.9, 0.95, "read from input", color = "orange", horizontalalignment = 'right')
    fig.text(0.5, 0.92, "multiple modules running", color = "blue", horizontalalignment = 'center')
    print "> ... Saving to file: '{}'".format(pdfFile)
    plt.savefig(pdfFile)

#=======================================
if __name__=="__main__":
    import sys

    argc = len(sys.argv)
    if argc not in [2,3]:
        sys.stderr.write("\n\033[1mERROR:\033[0m Wrong number of arguments specified ({}).  Should be 2 or 3.\n\n".format(argc))
        printHelp()
        exit(0)

    doGraphic = False
    pdfFile="stall.pdf"
    if argc == 3:
        arg = sys.argv[1]
        if arg == '-g':
            doGraphic = True
        elif arg.find("-g=") != -1:
            doGraphic = True
            pdfFile = arg.split('=')[1]
            if not re.match(r'^[\w\.]+$', pdfFile):
                print "Malformed file name '{}' supplied with the '-g' option.".format(pdfFile)
                print "Only characters 0-9, a-z, A-Z, '_', and '.' are allowed."
                exit(1)
        else:
            print "Unknown argument ",sys.argv[1]
            exit(1)

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
        createPDFImage(pdfFile, processingSteps, numStreams, stalledModules)
    printStalledModulesInOrder(stalledModules)
