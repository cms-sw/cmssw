#!/usr/bin/env python3
from __future__ import print_function
from builtins import range
from itertools import groupby
from operator import attrgetter,itemgetter
import sys
import json
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


kMicroToSec = 0.000001
#Special names
kSourceFindEvent = "sourceFindEvent"
kSourceDelayedRead ="sourceDelayedRead"
#this value is defined in the framework itself
kLargestLumiNumber = 4294967295

#these values must match the enum class Phase in tracer_setupFile.cc
class Phase (object):
  destruction = -15
  endJob = -12
  endStream = -11
  writeProcessBlock = -10
  endProcessBlock = -9
  globalWriteRun = -7
  globalEndRun = -6
  streamEndRun = -4
  globalWriteLumi = -4
  globalEndLumi = -3
  streamEndLumi = -2
  clearEvent = -1
  Event = 0
  streamBeginLumi = 2
  globalBeginLumi = 3
  streamBeginRun = 5
  globalBeginRun = 6
  accessInputProcessBlock = 8
  beginProcessBlock = 9
  openFile = 10
  beginStream = 11
  beginJob  = 12
  esSync = 13
  esSyncEnqueue = 14
  getNextTransition = 15
  construction = 16
  startTracing = 17

#used for json output
class Activity (object):
  prefetch = 0
  acquire = 1
  process = 2
  delayedGet = 3

transitionToNames_ = {
    Phase.startTracing: 'start tracing',
    Phase.construction: 'construction',
    Phase.destruction: 'destruction',
    Phase.beginJob: 'begin job',
    Phase.endJob: 'end job',
    Phase.beginStream: 'begin stream',
    Phase.endStream: 'end stream',
    Phase.beginProcessBlock: 'begin process block',
    Phase.endProcessBlock: 'end process block',
    Phase.accessInputProcessBlock: 'access input process block',
    Phase.writeProcessBlock: 'write process block',
    Phase.globalBeginRun: 'global begin run',
    Phase.globalEndRun: 'global end run',
    Phase.globalWriteRun: 'global write run',
    Phase.streamBeginRun: 'stream begin run',
    Phase.streamEndRun: 'stream end run',
    Phase.globalBeginLumi: 'global begin lumi',
    Phase.globalEndLumi: 'global end lumi',
    Phase.globalWriteLumi: 'global write lumi',
    Phase.streamBeginLumi: 'stream begin lumi',
    Phase.streamEndLumi: 'stream end lumi',
    Phase.esSyncEnqueue: 'EventSetup synchronization',
    Phase.esSync: 'EventSetup synchronization',
    Phase.Event: 'event',
    Phase.clearEvent: 'clear event',
    Phase.getNextTransition: 'get next transition'
}

def transitionName(transition):
    return transitionToNames_[transition]

transitionToIndent_ = {
    Phase.startTracing: 0,
    Phase.construction: 0,
    Phase.destruction: 0,
    Phase.endJob: 0,
    Phase.beginJob: 0,
    Phase.beginStream: 0,
    Phase.endStream: 0,
    Phase.beginProcessBlock: 1,
    Phase.endProcessBlock: 1,
    Phase.accessInputProcessBlock: 1,
    Phase.writeProcessBlock: 1,
    Phase.globalBeginRun: 1,
    Phase.globalEndRun: 1,
    Phase.globalWriteRun: 1,
    Phase.streamBeginRun: 1,
    Phase.streamEndRun: 1,
    Phase.globalBeginLumi: 2,
    Phase.globalEndLumi: 2,
    Phase.globalWriteLumi: 2,
    Phase.streamBeginLumi: 2,
    Phase.streamEndLumi: 2,
    Phase.Event: 3,
    Phase.clearEvent: 3,
    Phase.esSyncEnqueue: 1,
    Phase.esSync: 1,
    Phase.getNextTransition: 1
}
def transitionIndentLevel(transition):
    return transitionToIndent_[transition]

globalTransitions_ = {
    Phase.startTracing,
    Phase.construction,
    Phase.destruction,
    Phase.endJob,
    Phase.beginJob,
    Phase.beginProcessBlock,
    Phase.endProcessBlock,
    Phase.accessInputProcessBlock,
    Phase.writeProcessBlock,
    Phase.globalBeginRun,
    Phase.globalEndRun,
    Phase.globalWriteRun,
    Phase.globalBeginLumi,
    Phase.globalEndLumi,
    Phase.globalWriteLumi,
    Phase.esSyncEnqueue,
    Phase.esSync,
    Phase.getNextTransition
}
def transitionIsGlobal(transition):
    return transition in globalTransitions_;

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

def findMatchingTransition(sync, containers):
    for i in range(len(containers)):
        if containers[i][-1]["sync"] == sync:
            return i
    #need more exhausting search
    for i in range(len(containers)):
        for t in containers[i]:
            if t["sync"] == sync:
                return i

    print("find failed",sync, containers)
    return None

def popQueuedTransitions(sync, container):
    results = []
    for i in range(len(container)):
        if sync == container[i]["sync"]:
            results.append(container[i])
            results.append(container[i+1])
            del container[i]
            del container[i]
            break
    return results
        
transitionsToFindMatch_ = {
    Phase.globalEndRun,
    Phase.globalEndLumi,
    Phase.globalWriteRun,
    Phase.globalWriteLumi
}

class PreFrameworkTransitionParser (FrameworkTransitionParser):
    def __init__(self, payload):
        super().__init__(payload)
    def textSpecial(self):
        return "starting"
    def jsonInfo(self, counter, data):
        if transitionIsGlobal(self.transition):
            index = 0
            if self.transition == Phase.startTracing:
                data["globals"][0].append(jsonTransition(type=self.transition, id=index, sync=list(self.sync),start=0, finish=self.time ))
                return
            elif self.transition == Phase.esSync:
                if self.sync[1] == kLargestLumiNumber:
                    #at end run transition
                    index = findMatchingTransition(list(self.sync), data["globals"])
                    container = data['globals'][index]
                    container[-1]["finish"] = self.time*kMicroToSec
                else:
                    data['queued'][-1]["finish"] = self.time*kMicroToSec
                    data['queued'].append( jsonTransition(type=self.transition, id = index, sync=list(self.sync), start=self.time , finish=0))
                    return
            elif self.transition==Phase.globalBeginRun:
                index = self.index
                #find associated es queued items
                queued = data["queued"]
                q = popQueuedTransitions(list(self.sync), queued)
                globals = data['globals']
                while index+1 > len(globals):
                    globals.append([])
                container = globals[index]
                #find source, should be previous
                last = container[-1]
                if last["type"]==Phase.globalBeginRun and last["isSrc"]:
                    last["sync"]=list(self.sync)
                container.append(q[0])
                container.append(q[1])
            elif self.transition==Phase.globalBeginLumi:
                index = self.index
                #find associated es queued items
                queued = data["queued"]
                q = popQueuedTransitions(list(self.sync), queued)
                globals = data['globals']
                while index+1 > len(globals):
                    globals.append([])
                container = globals[index]
                #find source, should be previous
                last = container[-1]
                if last["type"]==Phase.globalBeginLumi and last["isSrc"]:
                    last["sync"]=list(self.sync)
                container.append(q[0])
                container.append(q[1])
            elif self.transition in transitionsToFindMatch_:
                index = findMatchingTransition(list(self.sync), data["globals"])
            globals = data["globals"]
            while index+1 > len(globals):
                globals.append([])
            container = globals[index]
        else:
            streams = data["streams"]
            while len(streams) < self.index+1:
                streams.append([])
            container = streams[self.index]
            if self.transition == Phase.Event:
                #find source, should be previous
                last = container[-1]
                if last["type"]==Phase.Event and last["isSrc"]:
                    last["sync"]=list(self.sync)
            index = self.index
        container.append( jsonTransition(type=self.transition, id = index, sync=list(self.sync), start=self.time , finish=0))
        

class PostFrameworkTransitionParser (FrameworkTransitionParser):
    def __init__(self, payload):
        super().__init__(payload)
    def textSpecial(self):
        return "finished"
    def jsonInfo(self, counter, data):
        if transitionIsGlobal(self.transition):
            if self.transition == Phase.esSync and self.sync[1] != kLargestLumiNumber:
                data['queued'][-1]['finish']=self.time*kMicroToSec
                return
            index = findMatchingTransition(list(self.sync), data["globals"])
            container = data["globals"][index]
        else:
            container = data["streams"][self.index]
        container[-1]["finish"]=self.time*kMicroToSec


class QueuingFrameworkTransitionParser (FrameworkTransitionParser):
    def __init__(self, payload):
        super().__init__(payload)
    def textSpecial(self):
        return "queuing"
    def jsonInfo(self, counter, data):
        index = -1
        if self.sync[1] == kLargestLumiNumber:
            #find the mtching open run
            index = findMatchingTransition([self.sync[0],0,0], data["globals"])
            data["globals"][index].append( jsonTransition(type=self.transition, id = index, sync=list(self.sync), start=self.time , finish=0))
        else:
            data["queued"].append(jsonTransition(type=self.transition, id = index, sync=list(self.sync), start=self.time , finish=0))

class SourceTransitionParser(object):
    def __init__(self, payload):
        self.transition = int(payload[0])
        if self.transition == Phase.getNextTransition:
            self.time = int(payload[1])
            self.index = -1
            return
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
        if self.transition == Phase.getNextTransition:
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
    def jsonInfo(self, counter, data):
        if self.transition == Phase.construction:
            index = counter.start()
            container = data["globals"]
        elif self.transition == Phase.getNextTransition:
            data['nextTrans'].append(jsonTransition(type=self.transition, id=self.index, sync=[0,0,0], start=self.time, finish=0, isSrc=True))
            return
        elif self.transition == Phase.Event:
            index = self.index
            container = data["streams"]
        else:
            container = data["globals"]
            index = self.index
        while len(container) < index+1:
            container.append([])
        nextTrans = data['nextTrans']
        if nextTrans:
            data['nextTrans'] = []
            for t in nextTrans:
                t['id']=index
                #find proper time order in the container
                transStartTime = t['start']
                inserted = False
                for i in range(-1, -1*len(container[index]), -1):
                    if transStartTime > container[index][i]['start']:
                        if i == -1:
                            container[index].append(t)
                            inserted = True
                            break
                        else:
                            container[index].insert(i+1,t)
                            inserted = True
                            break
                if not inserted:
                    container[index].insert(0,t)
        container[index].append(jsonTransition(type=self.transition, id=index, sync=[0,0,0], start=self.time, finish=0, isSrc=True))

class PostSourceTransitionParser(SourceTransitionParser):
    def __init__(self, payload):
        super().__init__(payload)
    def textSpecial(self):
        return "finished"
    def jsonInfo(self, counter, data):
        if self.transition == Phase.Event:
            container = data["streams"]
        elif self.transition == Phase.getNextTransition:
            data['nextTrans'][-1]['finish'] = self.time*kMicroToSec
            return
        elif self.transition == Phase.construction:
            container = data["globals"]
            pre = None
            for i, g in enumerate(data['globals']):
                for t in reversed(g):
                    if t["type"] != Phase.construction:
                        break
                    if t["isSrc"]:
                        pre = t
                        break
                if pre:
                    pre["finish"]=self.time*kMicroToSec
                    break
            counter.finish(i)
            return
        else:
            container = data["globals"]
        index = self.index

        container[index][-1]["finish"]=self.time*kMicroToSec

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
    def _preJson(self, activity, counter, data):
        if transitionIsGlobal(self.transition):
            container = data["modGlobals"]
        else:
            container = data["modStreams"]
        index = self.index
        while index+1 > len(container):
            container.append([[]])
        container = container[index]
        #find open slot
        foundOpenSlot = False
        for slot in container:
            if len(slot) == 0:
                foundOpenSlot = True
                break
            if slot[-1]["finish"] != 0:
                foundOpenSlot = True
                break
        if not foundOpenSlot:
            container.append([])
            slot = container[-1]
        slot.append(jsonModuleTransition(type=self.transition, id=self.index, modID=self.moduleID, activity=activity, start=self.time))
        return slot[-1]
    def _postJson(self, counter, data):
        if transitionIsGlobal(self.transition):
            container = data["modGlobals"]
        else:
            container = data["modStreams"]
        index = self.index
        container = container[index]
        #find slot containing the pre
        for slot in container:
            if slot[-1]["mod"] == self.moduleID:
                slot[-1]["finish"]=self.time*kMicroToSec
                return
        print(f"failed to find {self.moduleID} for {self.transition} in {self.index} with {container}")

class PreEDModuleTransitionParser(EDModuleTransitionParser):
    def __init__(self, payload, names):
        super().__init__(payload, names)
    def textSpecial(self):
        return "starting action"
    def jsonInfo(self, counter, data):
        return self._preJson(Activity.process, counter,data)

class PostEDModuleTransitionParser(EDModuleTransitionParser):
    def __init__(self, payload, names):
        super().__init__(payload, names)
    def textSpecial(self):
        return "finished action"
    def jsonInfo(self, counter, data):
        return self._postJson(counter,data)
        
class PreEDModulePrefetchingParser(EDModuleTransitionParser):
    def __init__(self, payload, names):
        super().__init__(payload, names)
    def textSpecial(self):
        return "starting prefetch"
    def jsonInfo(self, counter, data):
        #the total time in prefetching isn't useful, but seeing the start is
        kPrefetchLength = 2*kMicroToSec
        entry = self._preJson(Activity.prefetch, counter,data)
        entry["finish"]=entry["start"]+kPrefetchLength
        return entry


class PostEDModulePrefetchingParser(EDModuleTransitionParser):
    def __init__(self, payload, names):
        super().__init__(payload, names)
    def textSpecial(self):
        return "finished prefetch"
    def jsonInfo(self, counter, data):
        pass

class PreEDModuleAcquireParser(EDModuleTransitionParser):
    def __init__(self, payload, names):
        super().__init__(payload, names)
    def textSpecial(self):
        return "starting acquire"
    def jsonInfo(self, counter, data):
        return self._preJson(Activity.acquire, counter,data)

class PostEDModuleAcquireParser(EDModuleTransitionParser):
    def __init__(self, payload, names):
        super().__init__(payload, names)
    def textSpecial(self):
        return "finished acquire"
    def jsonInfo(self, counter, data):
        return self._postJson(counter,data)

class PreEDModuleEventDelayedGetParser(EDModuleTransitionParser):
    def __init__(self, payload, names):
        super().__init__(payload, names)
    def textSpecial(self):
        return "starting delayed get"
    def jsonInfo(self, counter, data):
        return self._preJson(Activity.delayedGet, counter,data)

class PostEDModuleEventDelayedGetParser(EDModuleTransitionParser):
    def __init__(self, payload, names):
        super().__init__(payload, names)
    def textSpecial(self):
        return "finished delayed get"
    def jsonInfo(self, counter, data):
        return self._postJson(counter,data)

class PreEventReadFromSourceParser(EDModuleTransitionParser):
    def __init__(self, payload, names):
        super().__init__(payload, names)
    def textSpecial(self):
        return "starting read from source"
    def jsonInfo(self, counter, data):
        slot = self._preJson(Activity.process, counter,data)
        slot['isSrc'] = True
        return slot

class PostEventReadFromSourceParser(EDModuleTransitionParser):
    def __init__(self, payload, names):
        super().__init__(payload, names)
    def textSpecial(self):
        return "finished read from source"
    def jsonInfo(self, counter, data):
        return self._postJson(counter,data)

class ESModuleTransitionParser(object):
    def __init__(self, payload, moduleNames, esModuleNames, recordNames):
        self.transition = int(payload[0])
        self.index = int(payload[1])
        self.moduleID = int(payload[2])
        self.moduleName = esModuleNames[self.moduleID]
        self.recordID = int(payload[3])
        self.recordName = recordNames[self.recordID]
        self.callID = int(payload[4])
        self.requestingModuleID = int(payload[5])
        self.requestingModuleName = None
        if self.requestingModuleID < 0 :
            self.requestingModuleName = esModuleNames[-1*self.requestingModuleID]
        else:
            self.requestingModuleName = moduleNames[self.requestingModuleID]
        self.time = int(payload[6])
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
    def _preJson(self, activity, counter, data):
        if transitionIsGlobal(self.transition):
            container = data["modGlobals"]
        else:
            container = data["modStreams"]
        index = self.index
        while index+1 > len(container):
            container.append([[]])
        container = container[index]
        #find open slot
        foundOpenSlot = False
        for slot in container:
            if len(slot) == 0:
                foundOpenSlot = True
                break
            if slot[-1]["finish"] != 0:
                foundOpenSlot = True
                break
        if not foundOpenSlot:
            container.append([])
            slot = container[-1]
        slot.append(jsonModuleTransition(type=self.transition, id=self.index, modID=-1*self.moduleID, activity=activity, start=self.time))
        slot[-1]['callID']=self.callID
        return slot[-1]
    def _postJson(self, counter, data):
        if transitionIsGlobal(self.transition):
            container = data["modGlobals"]
        else:
            container = data["modStreams"]
        index = self.index
        container = container[index]
        #find slot containing the pre
        for slot in container:
            if slot[-1]["mod"] == -1*self.moduleID and slot[-1].get('callID',0) == self.callID:
                slot[-1]["finish"]=self.time*kMicroToSec
                del slot[-1]['callID']
                return
        print(f"failed to find {-1*self.moduleID} for {self.transition} in {self.index} with {container}")


class PreESModuleTransitionParser(ESModuleTransitionParser):
    def __init__(self, payload, names, esNames, recordNames):
        super().__init__(payload, names, esNames, recordNames)
    def textSpecial(self):
        return "starting action"
    def jsonInfo(self, counter, data):
        return self._preJson(Activity.process, counter,data)

class PostESModuleTransitionParser(ESModuleTransitionParser):
    def __init__(self, payload, names, esNames, recordNames):
        super().__init__(payload, names, esNames, recordNames)
    def textSpecial(self):
        return "finished action"
    def jsonInfo(self, counter, data):
        return self._postJson(counter,data)

class PreESModulePrefetchingParser(ESModuleTransitionParser):
    def __init__(self, payload, names, esNames, recordNames):
        super().__init__(payload, names, esNames, recordNames)
    def textSpecial(self):
        return "starting prefetch"
    def jsonInfo(self, counter, data):
        entry = self._preJson(Activity.prefetch, counter,data)
        entry["finish"] = entry["start"]+2*kMicroToSec;
        return entry

class PostESModulePrefetchingParser(ESModuleTransitionParser):
    def __init__(self, payload, names, esNames, recordNames):
        super().__init__(payload, names, esNames, recordNames)
    def textSpecial(self):
        return "finished prefetch"
    def jsonInfo(self, counter, data):
        pass

class PreESModuleAcquireParser(ESModuleTransitionParser):
    def __init__(self, payload, names, recordNames):
        super().__init__(payload, names, recordNames)
    def textSpecial(self):
        return "starting acquire"
    def jsonInfo(self, counter, data):
        return self._preJson(Activity.acquire, counter,data)

class PostESModuleAcquireParser(ESModuleTransitionParser):
    def __init__(self, payload, names, esNames, recordNames):
        super().__init__(payload, names, esNames, recordNames)
    def textSpecial(self):
        return "finished acquire"
    def jsonInfo(self, counter, data):
        return self._postJson(counter,data)


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
    
class Counter(object):
    def __init__(self):
        self.activeSlots = [False]
    def start(self):
        if 0 != self.activeSlots.count(False):
            index = self.activeSlots.index(False)
            self.activeSlots[index]=True
            return index
        index = len(self.activeSlots)
        self.activeSlots.append(True)
        return  index
    def finish(self, index):
        self.activeSlots[index] = False


def jsonTransition(type, id, sync, start, finish, isSrc=False):
    return {"type": type, "id": id, "sync": sync, "start": start*kMicroToSec, "finish": finish*kMicroToSec, "isSrc":isSrc}

def jsonModuleTransition(type, id, modID, activity, start, finish=0):
    return {"type": type, "id": id, "mod": modID, "act": activity, "start": start*kMicroToSec, "finish": finish*kMicroToSec}

def startTime(x):
    return x["start"]
def jsonInfo(parser):
    counter = Counter()
    data = {"globals": [[]], "streams" :[[]], "queued": [], "nextTrans": []}
    if not parser._frameworkOnly:
        data["modGlobals"] = [[]]
        data["modStreams"] = [[[]]]
    for p in parser.processingSteps():
        p.jsonInfo(counter, data)
    #make sure everything is sorted
    for g in data["globals"]:
        g.sort(key=startTime)
    del data["queued"]
    del data['nextTrans']
    final = {"transitions" : [] , "modules": [], "esModules": []}
    final["transitions"].append({ "name":"Global", "slots": []})
    globals = final["transitions"][-1]["slots"]
    for i, g in enumerate(data["globals"]):
        globals.append(g)
        if len(data["modGlobals"]) < i+1:
            break
        for mod in data["modGlobals"][i]:
            globals.append(mod)
    for i,s in enumerate(data["streams"]):
        final["transitions"].append({"name": f"Stream {i}", "slots":[]})
        stream = final["transitions"][-1]["slots"]
        stream.append(s)
        for mod in data["modStreams"][i]:
            stream.append(mod)


    if not parser._frameworkOnly:
        max = 0
        for k in parser._moduleNames.keys():
            if k > max:
                max = k
        
        final["modules"] =['']*(max+1)
        final["modules"][0] = 'source'
        for k,v in parser._moduleNames.items():
            final["modules"][k]=v
        
        max = 0
        for k in parser._esModuleNames.keys():
            if k > max:
                max = k
        final["esModules"] = ['']*(max+1)
        for k,v in parser._esModuleNames.items():
            final["esModules"][k] = v
    return final
    
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
    parser.add_argument('-j', '--json',
                        action='store_true',
                        help='''Write output in json format.''' )
    parser.add_argument('-w', '--web',
                        action='store_true',
                        help='''Writes data.js file that can be used with the web based inspector. To use, copy directory ${CMSSW_RELEASE_BASE}/src/FWCore/Services/template/web to a web accessible area and move data.js into that directory.''')
    
    args = parser.parse_args()

    parser = TracerCompactFileParser(args.filename, args.frameworkOnly)
    if args.json or args.web:
        j = json.dumps(jsonInfo(parser))
        if args.json:
            print(j)
        if args.web:
            j ='export const data = ' + j
            f=open('data.js', 'w')
            f.write(j)
            f.close()
    else:
        textOutput(parser)
