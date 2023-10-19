#!/usr/bin/env python3
from __future__ import print_function
from builtins import range
from itertools import groupby
from operator import attrgetter,itemgetter
import sys
from collections import defaultdict
#----------------------------------------------
def printHelp():
    s = '''
To Use: Add the Tracer Service to the cmsRun job use something like this
  in the configuration:

  process.add_(cms.Service("Tracer", fileName = cms.untracked.string("tracer.log")))

  After running the job, execute this script and pass the name of the
  Tracer log file to the script.

  This script will output a more human readable form of the data in the Tracer log file.'''
    return s

#these values come from tracer_setupFile.cc
#enum class Step : char {
#  preSourceTransition = 'S',
#  postSourceTransition = 's',
#  preModulePrefetching = 'P',
#  postModulePrefetching = 'p',
#  preModuleEventAcquire = 'A',
#  postModuleEventAcquire = 'a',
#  preModuleTransition = 'M',
#  preEventReadFromSource = 'R',
#  postEventReadFromSource = 'r',
#  preModuleEventDelayedGet = 'D',
#  postModuleEventDelayedGet = 'd',
#  postModuleTransition = 'm',
#  preESModulePrefetching = 'Q',
#  postESModulePrefetching = 'q',
#  preESModule = 'N',
#  postESModule = 'n',
#  preESModuleAcquire = 'B',
#  postESModuleAcquire = 'b',
#  preFrameworkTransition = 'F',
#  postFrameworkTransition = 'f'
#};


#Special names
kSourceFindEvent = "sourceFindEvent"
kSourceDelayedRead ="sourceDelayedRead"

#these values must match the enum class Phase in tracer_setupFile.cc
class Phase (object):
  destruction = -14
  endJob = -11
  endStream = -10
  writeProcessBlock = -9
  endProcessBlock = -8
  globalWriteRun = -6
  globalEndRun = -5
  streamEndRun = -4
  globalWriteLumi = -3
  globalEndLumi = -2
  streamEndLumi = -1
  Event = 0
  streamBeginLumi = 1
  globalBeginLumi = 2
  streamBeginRun = 4
  globalBeginRun = 5
  accessInputProcessBlock = 7
  beginProcessBlock = 8
  beginStream = 10
  beginJob  = 11
  esSync = 12
  esSyncEnqueue = 13
  construction = 14
  startTracing = 15


def transitionName(transition):
    if transition == Phase.startTracing:
        return 'start tracing'
    if transition == Phase.construction:
        return 'construction'
    if transition == Phase.destruction:
        return 'destruction'
    if transition == Phase.beginJob:
        return 'begin job'
    if transition == Phase.endJob:
        return 'end job'
    if transition == Phase.beginStream:
        return 'begin stream'
    if transition == Phase.endStream:
        return 'end stream'
    if transition == Phase.beginProcessBlock:
        return 'begin process block'
    if transition == Phase.endProcessBlock:
        return 'end process block'
    if transition == Phase.accessInputProcessBlock:
        return 'access input process block'
    if transition == Phase.writeProcessBlock:
        return 'write process block'
    if transition == Phase.globalBeginRun:
        return 'global begin run'
    if transition == Phase.globalEndRun:
        return 'global end run'
    if transition == Phase.globalWriteRun:
        return 'global write run'
    if transition == Phase.streamBeginRun:
        return 'stream begin run'
    if transition == Phase.streamEndRun:
        return 'stream end run'
    if transition == Phase.globalBeginLumi:
        return 'global begin lumi'
    if transition == Phase.globalEndLumi:
        return 'global end lumi'
    if transition == Phase.globalWriteLumi:
        return 'global write lumi'
    if transition == Phase.streamBeginLumi:
        return 'stream begin lumi'
    if transition == Phase.streamEndLumi:
        return 'stream end lumi'
    if transition == Phase.esSyncEnqueue:
        return 'EventSetup synchronization'
    if transition == Phase.esSync:
        return 'EventSetup synchronization'
    if transition == Phase.Event:
        return 'event'

def transitionIndentLevel(transition):
    if transition == Phase.startTracing:
        return 0
    if transition == Phase.construction or transition == Phase.destruction:
        return 0
    if transition == Phase.endJob or transition == Phase.beginJob:
        return 0
    if transition == Phase.beginStream or transition == Phase.endStream:
        return 0
    if transition == Phase.beginProcessBlock or transition == Phase.endProcessBlock:
        return 1
    if transition == Phase.accessInputProcessBlock:
        return 1
    if transition == Phase.writeProcessBlock:
        return 1
    if transition == Phase.globalBeginRun or Phase.globalEndRun == transition:
        return 1
    if transition == Phase.globalWriteRun:
        return 1
    if transition == Phase.streamBeginRun or Phase.streamEndRun == transition:
        return 1
    if transition == Phase.globalBeginLumi or Phase.globalEndLumi == transition:
        return 2
    if transition == Phase.globalWriteLumi:
        return 2
    if transition == Phase.streamBeginLumi or Phase.streamEndLumi == transition:
        return 2
    if transition == Phase.Event:
        return 3
    if transition == Phase.esSyncEnqueue or transition == Phase.esSync:
        return 1
    return None

def textPrefix_(time, indentLevel):
    #using 11 spaces for time should accomodate a job that runs 24 hrs
    return f'{time:>11} '+"++"*indentLevel

class FrameworkTransitionParser (object):
    def __init__(self, payload):
        self.transition = int(payload[0])
        self.index = int(payload[1])
        self.sync = (int(payload[2]), int(payload[3]), int(payload[4]))
        self.time = int(payload[5])
    def indentLevel(self):
        return transitionIndentLevel(self.transition)
    def textPrefix(self):
        return textPrefix_(self.time, self.indentLevel())
    def syncText(self):
        if self.transition == Phase.globalBeginRun or Phase.globalEndRun == self.transition:
            return f'run={self.sync[0]}'
        if self.transition == Phase.globalWriteRun:
            return f'run={self.sync[0]}'
        if self.transition == Phase.streamBeginRun or Phase.streamEndRun == self.transition:
            return f'run={self.sync[0]}'
        if self.transition == Phase.globalBeginLumi or Phase.globalEndLumi == self.transition:
            return f'run={self.sync[0]} lumi={self.sync[1]}'
        if self.transition == Phase.globalWriteLumi:
            return f'run={self.sync[0]} lumi={self.sync[1]}'
        if self.transition == Phase.streamBeginLumi or Phase.streamEndLumi == self.transition:
            return f'run={self.sync[0]} lumi={self.sync[1]}'
        if self.transition == Phase.Event:
            return f'run={self.sync[0]} lumi={self.sync[1]} event={self.sync[2]}'
        if self.transition == Phase.esSyncEnqueue or self.transition == Phase.esSync:
            return f'run={self.sync[0]} lumi={self.sync[1]}'
        if self.transition == Phase.beginJob:
            return ''
        if self.transition == Phase.beginProcessBlock or self.transition == Phase.endProcessBlock or self.transition == Phase.writeProcessBlock or self.transition == Phase.accessInputProcessBlock:
            return ''
        if self.transition == Phase.startTracing:
            return ''
        if self.transition == Phase.construction or self.transition == Phase.destruction:
            return ''
    def textPostfix(self):
        return f'{transitionName(self.transition)} : id={self.index} {self.syncText()}'
    def text(self, context):
        return f'{self.textPrefix()} {self.textSpecial()}: {self.textPostfix()}'

class PreFrameworkTransitionParser (FrameworkTransitionParser):
    def __init__(self, payload):
        super().__init__(payload)
    def textSpecial(self):
        return "starting"
        

class PostFrameworkTransitionParser (FrameworkTransitionParser):
    def __init__(self, payload):
        super().__init__(payload)
    def textSpecial(self):
        return "finished"

class QueuingFrameworkTransitionParser (FrameworkTransitionParser):
    def __init__(self, payload):
        super().__init__(payload)
    def textSpecial(self):
        return "queuing"

class SourceTransitionParser(object):
    def __init__(self, payload):
        self.transition = int(payload[0])
        self.index = int(payload[1])
        self.time = int(payload[2])
    def indentLevel(self):
        if self.transition == Phase.globalBeginRun:
            return 1
        if self.transition == Phase.globalBeginLumi:
            return 2
        if self.transition == Phase.Event:
            return 3
        if self.transition == Phase.construction:
            return 1
        return None
    def textPrefix(self):
        return textPrefix_(self.time, self.indentLevel())
    def textPostfix(self):
        return f'source during {transitionName(self.transition)} : id={self.index}'
    def text(self, context):
        return f'{self.textPrefix()} {self.textSpecial()}: {self.textPostfix()}'

class PreSourceTransitionParser(SourceTransitionParser):
    def __init__(self, payload):
        super().__init__(payload)
    def textSpecial(self):
        return "starting"

class PostSourceTransitionParser(SourceTransitionParser):
    def __init__(self, payload):
        super().__init__(payload)
    def textSpecial(self):
        return "finished"

class EDModuleTransitionParser(object):
    def __init__(self, payload, moduleNames):
        self.transition = int(payload[0])
        self.index = int(payload[1])
        self.moduleID = int(payload[2])
        self.moduleName = moduleNames[self.moduleID]
        self.requestingModuleID = int(payload[3])
        self.requestingModuleName = None
        if self.requestingModuleID != 0:
            self.requestingModuleName = moduleNames[self.requestingModuleID]
        self.time = int(payload[4])
    def baseIndentLevel(self):
        return transitionIndentLevel(self.transition)
    def textPrefix(self, context):
        indent = 0
        if self.requestingModuleID != 0:
            indent = context[(self.transition, self.index, self.requestingModuleID)]
        context[(self.transition, self.index, self.moduleID)] = indent+1
        return textPrefix_(self.time, indent+1+self.baseIndentLevel())
    def textPostfix(self):
        return f'{self.moduleName} during {transitionName(self.transition)} : id={self.index}'
    def text(self, context):
        return f'{self.textPrefix(context)} {self.textSpecial()}: {self.textPostfix()}'

class PreEDModuleTransitionParser(EDModuleTransitionParser):
    def __init__(self, payload, names):
        super().__init__(payload, names)
    def textSpecial(self):
        return "starting action"

class PostEDModuleTransitionParser(EDModuleTransitionParser):
    def __init__(self, payload, names):
        super().__init__(payload, names)
    def textSpecial(self):
        return "finished action"

class PreEDModulePrefetchingParser(EDModuleTransitionParser):
    def __init__(self, payload, names):
        super().__init__(payload, names)
    def textSpecial(self):
        return "starting prefetch"

class PostEDModulePrefetchingParser(EDModuleTransitionParser):
    def __init__(self, payload, names):
        super().__init__(payload, names)
    def textSpecial(self):
        return "finished prefetch"

class PreEDModuleAcquireParser(EDModuleTransitionParser):
    def __init__(self, payload, names):
        super().__init__(payload, names)
    def textSpecial(self):
        return "starting acquire"

class PostEDModuleAcquireParser(EDModuleTransitionParser):
    def __init__(self, payload, names):
        super().__init__(payload, names)
    def textSpecial(self):
        return "finished acquire"

class PreEDModuleEventDelayedGetParser(EDModuleTransitionParser):
    def __init__(self, payload, names):
        super().__init__(payload, names)
    def textSpecial(self):
        return "starting delayed get"

class PostEDModuleEventDelayedGetParser(EDModuleTransitionParser):
    def __init__(self, payload, names):
        super().__init__(payload, names)
    def textSpecial(self):
        return "finished delayed get"

class PreEventReadFromSourceParser(EDModuleTransitionParser):
    def __init__(self, payload, names):
        super().__init__(payload, names)
    def textSpecial(self):
        return "starting read from source"

class PostEventReadFromSourceParser(EDModuleTransitionParser):
    def __init__(self, payload, names):
        super().__init__(payload, names)
    def textSpecial(self):
        return "finished read from source"

class ESModuleTransitionParser(object):
    def __init__(self, payload, moduleNames, esModuleNames, recordNames):
        self.transition = int(payload[0])
        self.index = int(payload[1])
        self.moduleID = int(payload[2])
        self.moduleName = esModuleNames[self.moduleID]
        self.recordID = int(payload[3])
        self.recordName = recordNames[self.recordID]
        self.requestingModuleID = int(payload[4])
        self.requestingModuleName = None
        if self.requestingModuleID < 0 :
            self.requestingModuleName = esModuleNames[-1*self.requestingModuleID]
        else:
            self.requestingModuleName = moduleNames[self.requestingModuleID]
        self.time = int(payload[5])
    def baseIndentLevel(self):
        return transitionIndentLevel(self.transition)
    def textPrefix(self, context):
        indent = 0
        indent = context[(self.transition, self.index, self.requestingModuleID)]
        context[(self.transition, self.index, -1*self.moduleID)] = indent+1
        return textPrefix_(self.time, indent+1+self.baseIndentLevel())
    def textPostfix(self):
        return f'esmodule {self.moduleName} in record {self.recordName} during {transitionName(self.transition)} : id={self.index}'
    def text(self, context):
        return f'{self.textPrefix(context)} {self.textSpecial()}: {self.textPostfix()}'

class PreESModuleTransitionParser(ESModuleTransitionParser):
    def __init__(self, payload, names, esNames, recordNames):
        super().__init__(payload, names, esNames, recordNames)
    def textSpecial(self):
        return "starting action"

class PostESModuleTransitionParser(ESModuleTransitionParser):
    def __init__(self, payload, names, esNames, recordNames):
        super().__init__(payload, names, esNames, recordNames)
    def textSpecial(self):
        return "finished action"

class PreESModulePrefetchingParser(ESModuleTransitionParser):
    def __init__(self, payload, names, esNames, recordNames):
        super().__init__(payload, names, esNames, recordNames)
    def textSpecial(self):
        return "starting prefetch"

class PostESModulePrefetchingParser(ESModuleTransitionParser):
    def __init__(self, payload, names, esNames, recordNames):
        super().__init__(payload, names, esNames, recordNames)
    def textSpecial(self):
        return "finished prefetch"

class PreESModuleAcquireParser(ESModuleTransitionParser):
    def __init__(self, payload, names, recordNames):
        super().__init__(payload, names, recordNames)
    def textSpecial(self):
        return "starting acquire"

class PostESModuleAcquireParser(ESModuleTransitionParser):
    def __init__(self, payload, names, esNames, recordNames):
        super().__init__(payload, names, esNames, recordNames)
    def textSpecial(self):
        return "finished acquire"


def lineParserFactory (step, payload, moduleNames, esModuleNames, recordNames, frameworkOnly):
    if step == 'F':
        parser = PreFrameworkTransitionParser(payload)
        if parser.transition == Phase.esSyncEnqueue:
            return QueuingFrameworkTransitionParser(payload)
        return parser
    if step == 'f':
        return PostFrameworkTransitionParser(payload)
    if step == 'S':
        return PreSourceTransitionParser(payload)
    if step == 's':
        return PostSourceTransitionParser(payload)
    if frameworkOnly:
        return None
    if step == 'M':
        return PreEDModuleTransitionParser(payload, moduleNames)
    if step == 'm':
        return PostEDModuleTransitionParser(payload, moduleNames)
    if step == 'P':
        return PreEDModulePrefetchingParser(payload, moduleNames)
    if step == 'p':
        return PostEDModulePrefetchingParser(payload, moduleNames)
    if step == 'A':
        return PreEDModuleAcquireParser(payload, moduleNames)
    if step == 'a':
        return PostEDModuleAcquireParser(payload, moduleNames)
    if step == 'D':
        return PreEDModuleEventDelayedGetParser(payload, moduleNames)
    if step == 'd':
        return PostEDModuleEventDelayedGetParser(payload, moduleNames)
    if step == 'R':
        return PreEventReadFromSourceParser(payload, moduleNames)
    if step == 'r':
        return PostEventReadFromSourceParser(payload, moduleNames)
    if step == 'N':
        return PreESModuleTransitionParser(payload, moduleNames, esModuleNames, recordNames)
    if step == 'n':
        return PostESModuleTransitionParser(payload, moduleNames, esModuleNames, recordNames)
    if step == 'Q':
        return PreESModulePrefetchingParser(payload, moduleNames, esModuleNames, recordNames)
    if step == 'q':
        return PostESModulePrefetchingParser(payload, moduleNames, esModuleNames, recordNames)
    if step == 'B':
        return PreESModuleAcquireParser(payload, moduleNames, esModuleNames, recordNames)
    if step == 'b':
        return PostESModuleAcquireParser(payload, moduleNames, esModuleNames, recordNames)

    
#----------------------------------------------
def processingStepsFromFile(f,moduleNames, esModuleNames, recordNames, frameworkOnly):
    for rawl in f:
        l = rawl.strip()
        if not l or l[0] == '#':
            continue
        (step,payload) = tuple(l.split(None,1))
        payload=payload.split()

        parser = lineParserFactory(step, payload, moduleNames, esModuleNames, recordNames, frameworkOnly)
        if parser:
            yield parser
    return

class TracerCompactFileParser(object):
    def __init__(self,f, frameworkOnly):
        streamBeginRun = str(Phase.streamBeginRun)
        numStreams = 0
        numStreamsFromSource = 0
        moduleNames = {}
        esModuleNames = {}
        recordNames = {}
        for rawl in f:
            l = rawl.strip()
            if l and l[0] == 'M':
                i = l.split(' ')
                if i[3] == streamBeginRun:
                    #found global begin run
                    numStreams = int(i[1])+1
                    break
            if numStreams == 0 and l and l[0] == 'S':
                s = int(l.split(' ')[1])
                if s > numStreamsFromSource:
                  numStreamsFromSource = s
            if len(l) > 5 and l[0:2] == "#M":
                (id,name)=tuple(l[2:].split())
                moduleNames[int(id)] = name
                continue
            if len(l) > 5 and l[0:2] == "#N":
                (id,name)=tuple(l[2:].split())
                esModuleNames[int(id)] = name
                continue
            if len(l) > 5 and l[0:2] == "#R":
                (id,name)=tuple(l[2:].split())
                recordNames[int(id)] = name
                continue

        self._f = f
        self._frameworkOnly = frameworkOnly
        if numStreams == 0:
          numStreams = numStreamsFromSource +2
        self.numStreams =numStreams
        self._moduleNames = moduleNames
        self._esModuleNames = esModuleNames
        self._recordNames = recordNames
        self.maxNameSize =0
        for n in moduleNames.items():
            self.maxNameSize = max(self.maxNameSize,len(n))
        for n in esModuleNames.items():
            self.maxNameSize = max(self.maxNameSize,len(n))
        self.maxNameSize = max(self.maxNameSize,len(kSourceDelayedRead))
        self.maxNameSize = max(self.maxNameSize, len('streamBeginLumi'))

    def processingSteps(self):
        """Create a generator which can step through the file and return each processing step.
        Using a generator reduces the memory overhead when parsing a large file.
            """
        self._f.seek(0)
        return processingStepsFromFile(self._f,self._moduleNames, self._esModuleNames, self._recordNames, self._frameworkOnly)

def textOutput( parser ):
    context = {}
    for p in parser.processingSteps():
        print(p.text(context))
    
#=======================================
if __name__=="__main__":
    import argparse
    import re
    import sys

    # Program options
    parser = argparse.ArgumentParser(description='Convert a compact tracer file into human readable output.',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=printHelp())
    parser.add_argument('filename',
                        type=argparse.FileType('r'), # open file
                        help='file to process')
    parser.add_argument('-f', '--frameworkOnly',
                        action='store_true',
                        help='''Output only the framework transitions, excluding the individual module transitions.''')
    args = parser.parse_args()

    parser = TracerCompactFileParser(args.filename, args.frameworkOnly)
    textOutput(parser)
