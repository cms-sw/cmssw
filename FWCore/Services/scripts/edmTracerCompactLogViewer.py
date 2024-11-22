#!/usr/bin/env python3
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
  destruction = -16
  endJob = -12
  endStream = -11
  writeProcessBlock = -10
  endProcessBlock = -9
  globalWriteRun = -7
  globalEndRun = -6
  streamEndRun = -5
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
  externalWork = 4
  temporary = 100

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
                data.indexedGlobal(0).append(jsonTransition(type=self.transition, id=index, sync=list(self.sync),start=0, finish=self.time ))
                return
            elif self.transition == Phase.esSync:
                if self.sync[1] == kLargestLumiNumber:
                    #at end run transition
                    index = findMatchingTransition(list(self.sync), data.allGlobals())
                    container = data.indexedGlobal(index)
                    container[-1]["finish"] = self.time*kMicroToSec
                else:
                    data._queued[-1]["finish"] = self.time*kMicroToSec
                    data._queued.append( jsonTransition(type=self.transition, id = index, sync=list(self.sync), start=self.time , finish=0))
                    return
            elif self.transition==Phase.globalBeginRun:
                index = self.index
                #find associated es queued items
                queued = data._queued
                q = popQueuedTransitions(list(self.sync), queued)
                container = data.indexedGlobal(index)
                #find source, should be previous
                last = container[-1]
                if last["type"]==Phase.globalBeginRun and last["isSrc"]:
                    last["sync"]=list(self.sync)
                container.append(q[0])
                container.append(q[1])
            elif self.transition==Phase.globalBeginLumi:
                index = self.index
                #find associated es queued items
                queued = data._queued
                q = popQueuedTransitions(list(self.sync), queued)
                container = data.indexedGlobal(index)
                #find source, should be previous
                last = container[-1]
                if last["type"]==Phase.globalBeginLumi and last["isSrc"]:
                    last["sync"]=list(self.sync)
                container.append(q[0])
                container.append(q[1])
            elif self.transition in transitionsToFindMatch_:
                index = findMatchingTransition(list(self.sync), data.allGlobals())
            container = data.indexedGlobal(index)
        else:
            container = data.indexedStream(self.index)
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
                data._queued[-1]['finish']=self.time*kMicroToSec
                return
            index = findMatchingTransition(list(self.sync), data.allGlobals())
            container = data.indexedGlobal(index)
        else:
            container = data.indexedStream(self.index)
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
            index = findMatchingTransition([self.sync[0],0,0], data.allGlobals())
            data.indexedGlobal(index).append( jsonTransition(type=self.transition, id = index, sync=list(self.sync), start=self.time , finish=0))
        else:
            data._queued.append(jsonTransition(type=self.transition, id = index, sync=list(self.sync), start=self.time , finish=0))

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
    def __init__(self, payload, moduleCentric):
        self._moduleCentric = moduleCentric
        super().__init__(payload)
    def textSpecial(self):
        return "starting"
    def jsonInfo(self, counter, data):
        if self.transition == Phase.getNextTransition:
            data._nextTrans.append(jsonTransition(type=self.transition, id=self.index, sync=[0,0,0], start=self.time, finish=0, isSrc=True))
            if self._moduleCentric:
                #this all goes to a module ID sorted container so not knowing actual index is OK
                data.findOpenSlotInModGlobals(0,0).append(data._nextTrans[-1])
            return
        elif self.transition == Phase.construction:
            index = counter.start()
            container = data.indexedGlobal(index)
        elif self.transition == Phase.Event:
            index = self.index
            container = data.indexedStream(index)
        else:
            index = self.index
            container = data.indexedGlobal(index)
        nextTrans = data._nextTrans
        if nextTrans:
            data._nextTrans = []
            for t in nextTrans:
                t['id']=index
                #find proper time order in the container
                transStartTime = t['start']
                inserted = False
                for i in range(-1, -1*len(container), -1):
                    if transStartTime > container[i]['start']:
                        if i == -1:
                            container.append(t)
                            inserted = True
                            break
                        else:
                            container.insert(i+1,t)
                            inserted = True
                            break
                if not inserted:
                    container.insert(0,t)
        container.append(jsonTransition(type=self.transition, id=index, sync=[0,0,0], start=self.time, finish=0, isSrc=True))
        if self._moduleCentric:
            if self.transition == Phase.Event:
                data.findOpenSlotInModStreams(index,0).append(container[-1])
            else:
                data.findOpenSlotInModGlobals(index,0).append(container[-1])

class PostSourceTransitionParser(SourceTransitionParser):
    def __init__(self, payload, moduleCentric):
        super().__init__(payload)
        self._moduleCentric = moduleCentric
    def textSpecial(self):
        return "finished"
    def jsonInfo(self, counter, data):
        index = self.index
        if self.transition == Phase.Event:
            container = data.indexedStream(index)
        elif self.transition == Phase.getNextTransition:
            data._nextTrans[-1]['finish'] = self.time*kMicroToSec
            return
        elif self.transition == Phase.construction:
            pre = None
            for i, g in enumerate(data.allGlobals()):
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
            container = data.indexedGlobal(index)

        container[-1]["finish"]=self.time*kMicroToSec

class EDModuleTransitionParser(object):
    def __init__(self, payload, moduleNames):
        self.transition = int(payload[0])
        self.index = int(payload[1])
        self.moduleID = int(payload[2])
        self.moduleName = moduleNames[self.moduleID]
        self.callID = int(payload[3])
        self.requestingModuleID = int(payload[4])
        self.requestingCallID = int(payload[5])
        self.requestingModuleName = None
        if self.requestingModuleID != 0:
            self.requestingModuleName = moduleNames[self.requestingModuleID]
        self.time = int(payload[6])
    def baseIndentLevel(self):
        return transitionIndentLevel(self.transition)
    def textPrefix(self, context):
        indent = 0
        if self.requestingModuleID != 0:
            indent = context[(self.transition, self.index, self.requestingModuleID, self.requestingCallID)]
        context[(self.transition, self.index, self.moduleID, self.callID)] = indent+1
        return textPrefix_(self.time, indent+1+self.baseIndentLevel())
    def textPostfix(self):
        return f'{self.moduleName} during {transitionName(self.transition)} : id={self.index}'
    def textIfTransform(self):
        if self.callID:
            return f' transform {self.callID-1}'
        return ''
    def text(self, context):
        return f'{self.textPrefix(context)} {self.textSpecial()}{self.textIfTransform()}: {self.textPostfix()}'
    def _preJson(self, activity, counter, data, mayUseTemp = False):
        index = self.index
        found = False
        if mayUseTemp:
            compare = lambda x: x['type'] == self.transition and x['id'] == self.index and x['mod'] == self.moduleID and x['call'] == self.callID and (x['act'] == Activity.temporary or x['act'] == Activity.externalWork)
            if transitionIsGlobal(self.transition):
                item,slot = data.findLastInModGlobals(index, self.moduleID, compare)
            else:
                item,slot = data.findLastInModStreams(index, self.moduleID, compare)
            if slot:
                if item['act'] == Activity.temporary:
                    slot.pop()
                else:
                    item['finish']=self.time*kMicroToSec
                found = True
        if not found:
            if transitionIsGlobal(self.transition):
                slot = data.findOpenSlotInModGlobals(index, self.moduleID)
            else:
                slot = data.findOpenSlotInModStreams(index, self.moduleID)
        slot.append(jsonModuleTransition(type=self.transition, id=self.index, modID=self.moduleID, callID=self.callID, activity=activity, start=self.time))
        return slot[-1]
    def _postJson(self, counter, data, injectAfter = None):
        compare = lambda x: x['id'] == self.index and x['mod'] == self.moduleID and x['call'] == self.callID and x['type'] == self.transition
        index = self.index
        if transitionIsGlobal(self.transition):
            item,slot = data.findLastInModGlobals(index, self.moduleID, compare)
        else:
            item,slot = data.findLastInModStreams(index, self.moduleID, compare)
        if item is None:
            print(f"failed to find {self.moduleID} for {self.transition} in {self.index}")
        else:
            item["finish"]=self.time*kMicroToSec
            if injectAfter:
                slot.append(injectAfter)

class PreEDModuleTransitionParser(EDModuleTransitionParser):
    def __init__(self, payload, names, moduleCentric):
        super().__init__(payload, names)
        self._moduleCentric = moduleCentric
    def textSpecial(self):
        return "starting action"
    def jsonInfo(self, counter, data):
        return self._preJson(Activity.process, counter,data, mayUseTemp=self._moduleCentric)

class PostEDModuleTransitionParser(EDModuleTransitionParser):
    def __init__(self, payload, names):
        super().__init__(payload, names)
    def textSpecial(self):
        return "finished action"
    def jsonInfo(self, counter, data):
        return self._postJson(counter,data)
        
class PreEDModulePrefetchingParser(EDModuleTransitionParser):
    def __init__(self, payload, names, moduleCentric):
        super().__init__(payload, names)
        self._moduleCentric = moduleCentric
    def textSpecial(self):
        return "starting prefetch"
    def jsonInfo(self, counter, data):
        #the total time in prefetching isn't useful, but seeing the start is
        entry = self._preJson(Activity.prefetch, counter,data)
        if self._moduleCentric:
            return entry
        kPrefetchLength = 2*kMicroToSec
        entry["finish"]=entry["start"]+kPrefetchLength
        return entry


class PostEDModulePrefetchingParser(EDModuleTransitionParser):
    def __init__(self, payload, names, moduleCentric):
        super().__init__(payload, names)
        self._moduleCentric = moduleCentric
    def textSpecial(self):
        return "finished prefetch"
    def jsonInfo(self, counter, data):
        if self._moduleCentric:
            #inject a dummy at end of the same slot to guarantee module run is in that slot
            return self._postJson(counter, data, jsonModuleTransition(type=self.transition, id=self.index, modID=self.moduleID, callID=self.callID, activity=Activity.temporary, start=self.time))
        pass

class PreEDModuleAcquireParser(EDModuleTransitionParser):
    def __init__(self, payload, names, moduleCentric):
        super().__init__(payload, names)
        self._moduleCentric = moduleCentric
    def textSpecial(self):
        return "starting acquire"
    def jsonInfo(self, counter, data):
        return self._preJson(Activity.acquire, counter,data, mayUseTemp=self._moduleCentric)

class PostEDModuleAcquireParser(EDModuleTransitionParser):
    def __init__(self, payload, names, moduleCentric):
        super().__init__(payload, names)
        self._moduleCentric = moduleCentric
    def textSpecial(self):
        return "finished acquire"
    def jsonInfo(self, counter, data):
        if self._moduleCentric:
            #inject an external work at end of the same slot to guarantee module run is in that slot
            return self._postJson(counter, data, jsonModuleTransition(type=self.transition, id=self.index, modID=self.moduleID, callID=self.callID, activity=Activity.externalWork, start=self.time))
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
        self.requestingCallID = int(payload[6])
        self.requestingModuleName = None
        if self.requestingModuleID < 0 :
            self.requestingModuleName = esModuleNames[-1*self.requestingModuleID]
        else:
            self.requestingModuleName = moduleNames[self.requestingModuleID]
        self.time = int(payload[7])
    def baseIndentLevel(self):
        return transitionIndentLevel(self.transition)
    def textPrefix(self, context):
        indent = 0
        indent = context[(self.transition, self.index, self.requestingModuleID, self.requestingCallID)]
        context[(self.transition, self.index, -1*self.moduleID, self.callID)] = indent+1
        return textPrefix_(self.time, indent+1+self.baseIndentLevel())
    def textPostfix(self):
        return f'esmodule {self.moduleName} in record {self.recordName} during {transitionName(self.transition)} : id={self.index}'
    def text(self, context):
        return f'{self.textPrefix(context)} {self.textSpecial()}: {self.textPostfix()}'
    def _preJson(self, activity, counter, data):
        index = self.index
        if transitionIsGlobal(self.transition):
            slot = data.findOpenSlotInModGlobals(index, -1*self.moduleID)
        else:
            slot = data.findOpenSlotInModStreams(index, -1*self.moduleID)
        slot.append(jsonModuleTransition(type=self.transition, id=self.index, modID=-1*self.moduleID, callID=self.callID, activity=activity, start=self.time))
        return slot[-1]
    def _postJson(self, counter, data):
        compare = lambda x: x['id'] == self.index and x['mod'] == -1*self.moduleID and x['call'] == self.callID
        index = self.index
        if transitionIsGlobal(self.transition):
            item,s = data.findLastInModGlobals(index, -1*self.moduleID, compare)
        else:
            item,s = data.findLastInModStreams(index, -1*self.moduleID, compare)
        if item is None:
            print(f"failed to find {-1*self.moduleID} for {self.transition} in {self.index}")
            return
        item["finish"]=self.time*kMicroToSec


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
    def __init__(self, payload, names, esNames, recordNames, moduleCentric):
        super().__init__(payload, names, esNames, recordNames)
        self._moduleCentric = moduleCentric
    def textSpecial(self):
        return "starting prefetch"
    def jsonInfo(self, counter, data):
        entry = self._preJson(Activity.prefetch, counter,data)
        if not self._moduleCentric:
            entry["finish"] = entry["start"]+2*kMicroToSec;
        return entry

class PostESModulePrefetchingParser(ESModuleTransitionParser):
    def __init__(self, payload, names, esNames, recordNames, moduleCentric):
        super().__init__(payload, names, esNames, recordNames)
        self._moduleCentric = moduleCentric
    def textSpecial(self):
        return "finished prefetch"
    def jsonInfo(self, counter, data):
        if self._moduleCentric:
            return self._postJson(counter, data)
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


def lineParserFactory (step, payload, moduleNames, esModuleNames, recordNames, frameworkOnly, moduleCentric):
    if step == 'F':
        parser = PreFrameworkTransitionParser(payload)
        if parser.transition == Phase.esSyncEnqueue:
            return QueuingFrameworkTransitionParser(payload)
        return parser
    if step == 'f':
        return PostFrameworkTransitionParser(payload)
    if step == 'S':
        return PreSourceTransitionParser(payload, moduleCentric)
    if step == 's':
        return PostSourceTransitionParser(payload, moduleCentric)
    if frameworkOnly:
        return None
    if step == 'M':
        return PreEDModuleTransitionParser(payload, moduleNames, moduleCentric)
    if step == 'm':
        return PostEDModuleTransitionParser(payload, moduleNames)
    if step == 'P':
        return PreEDModulePrefetchingParser(payload, moduleNames, moduleCentric)
    if step == 'p':
        return PostEDModulePrefetchingParser(payload, moduleNames, moduleCentric)
    if step == 'A':
        return PreEDModuleAcquireParser(payload, moduleNames, moduleCentric)
    if step == 'a':
        return PostEDModuleAcquireParser(payload, moduleNames, moduleCentric)
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
        return PreESModulePrefetchingParser(payload, moduleNames, esModuleNames, recordNames, moduleCentric)
    if step == 'q':
        return PostESModulePrefetchingParser(payload, moduleNames, esModuleNames, recordNames, moduleCentric)
    if step == 'B':
        return PreESModuleAcquireParser(payload, moduleNames, esModuleNames, recordNames)
    if step == 'b':
        return PostESModuleAcquireParser(payload, moduleNames, esModuleNames, recordNames)

    
#----------------------------------------------
def processingStepsFromFile(f,moduleNames, esModuleNames, recordNames, frameworkOnly, moduleCentric):
    for rawl in f:
        l = rawl.strip()
        if not l or l[0] == '#':
            continue
        (step,payload) = tuple(l.split(None,1))
        payload=payload.split()

        parser = lineParserFactory(step, payload, moduleNames, esModuleNames, recordNames, frameworkOnly, moduleCentric)
        if parser:
            yield parser
    return

class TracerCompactFileParser(object):
    def __init__(self,f, frameworkOnly, moduleCentric):
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
        self._moduleCentric = moduleCentric
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
        return processingStepsFromFile(self._f,self._moduleNames, self._esModuleNames, self._recordNames, self._frameworkOnly, self._moduleCentric)

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

class Containers(object):
    def __init__(self):
        self._modGlobals = [[]]
        self._modStreams = [[]]
        self._globals = [[]]
        self._streams = [[]]
        self._queued = []
        self._nextTrans = []
    def _extendIfNeeded(self, container, index):
        while len(container) < index+1:
            container.append([])
    def allGlobals(self):
        return self._globals
    def indexedGlobal(self, index):
        self._extendIfNeeded(self._globals, index)
        return self._globals[index]
    def allStreams(self):
        return self._streams
    def indexedStream(self, index):
        self._extendIfNeeded(self._streams, index)
        return self._streams[index]
    def _findOpenSlot(self, index, fullContainer):
        self._extendIfNeeded(fullContainer, index)
        container = fullContainer[index]
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
        return slot
    def findOpenSlotInModGlobals(self, index, modID):
        return self._findOpenSlot(index, self._modGlobals)
    def findOpenSlotInModStreams(self, index, modID):
        return self._findOpenSlot(index, self._modStreams)
    def _findLastIn(self, index, fullContainer, comparer):
        container = fullContainer[index]
        #find slot containing the pre
        for slot in container:
            if comparer(slot[-1]):
                return (slot[-1],slot)
        return (None,None)
    def findLastInModGlobals(self, index, modID, comparer):
        return self._findLastIn(index, self._modGlobals, comparer)
    def findLastInModStreams(self, index, modID, comparer):
        return self._findLastIn(index, self._modStreams, comparer)
    

class ModuleCentricContainers(object):
    def __init__(self):
        self._modules= []
        self._globals = [[]]
        self._streams = [[]]
        self._queued = []
        self._nextTrans = []
        self._moduleIDOffset = 0
    def _moduleID2Index(self, modID):
        return modID + self._moduleIDOffset
    def _extendIfNeeded(self, container, index):
        while len(container) < index+1:
            container.append([])
    def _extendModulesIfNeeded(self, container, index):
        while index + self._moduleIDOffset < 0:
            container.insert(0,[])
            self._moduleIDOffset +=1
        self._extendIfNeeded(container, self._moduleID2Index(index))
    def allGlobals(self):
        return self._globals
    def indexedGlobal(self, index):
        self._extendIfNeeded(self._globals, index)
        return self._globals[index]
    def allStreams(self):
        return self._streams
    def indexedStream(self, index):
        self._extendIfNeeded(self._streams, index)
        return self._streams[index]
    def _findOpenSlot(self, index, fullContainer):
        self._extendModulesIfNeeded(fullContainer, index)
        container = fullContainer[self._moduleID2Index(index)]
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
        return slot
    def findOpenSlotInModGlobals(self, index, modID):
        return self._findOpenSlot(modID, self._modules)
    def findOpenSlotInModStreams(self, index, modID):
        return self._findOpenSlot(modID, self._modules)
    def _findLastIn(self, index, fullContainer, comparer):
        if not fullContainer:
            return (None, None)
        if len(fullContainer) > self._moduleID2Index(index):
            container = fullContainer[self._moduleID2Index(index)]
        else:
            return (None, None)
        #find slot containing the pre
        for slot in container:
            if slot is not None and comparer(slot[-1]):
                return (slot[-1],slot)
        return (None, None)
    def findLastInModGlobals(self, index, modID, comparer):
        return self._findLastIn(modID, self._modules, comparer)
    def findLastInModStreams(self, index, modID, comparer):
        return self._findLastIn(modID, self._modules, comparer)

    

def jsonTransition(type, id, sync, start, finish, isSrc=False):
    return {"type": type, "id": id, "sync": sync, "start": start*kMicroToSec, "finish": finish*kMicroToSec, "isSrc":isSrc}

def jsonModuleTransition(type, id, modID, callID, activity, start, finish=0):
    return {"type": type, "id": id, "mod": modID, "call": callID, "act": activity, "start": start*kMicroToSec, "finish": finish*kMicroToSec}

def startTime(x):
    return x["start"]
def jsonInfo(parser):
    counter = Counter()
    if parser._moduleCentric:
        data = ModuleCentricContainers()
    else:
        data = Containers()
    for p in parser.processingSteps():
        p.jsonInfo(counter, data)
    #make sure everything is sorted
    for g in data.allGlobals():
        g.sort(key=startTime)
    final = {"transitions" : [] , "modules": [], "esModules": []}
    final["transitions"].append({ "name":"Global", "slots": []})
    globals = final["transitions"][-1]["slots"]
    for i, g in enumerate(data.allGlobals()):
        globals.append(g)
        if not parser._moduleCentric and not parser._frameworkOnly:
            if len(data._modGlobals) < i+1:
                break
            for mod in data._modGlobals[i]:
                globals.append(mod)
    for i,s in enumerate(data.allStreams()):
        final["transitions"].append({"name": f"Stream {i}", "slots":[]})
        stream = final["transitions"][-1]["slots"]
        stream.append(s)
        if not parser._moduleCentric and not parser._frameworkOnly:
            for mod in data._modStreams[i]:
                stream.append(mod)
    if parser._moduleCentric:
        sourceSlot = data._modules[data._moduleID2Index(0)]
        modules = []
        for i,m in parser._moduleNames.items():
            modules.append({"name": f"{m}", "slots":[]})
            slots = modules[-1]["slots"]
            foundSlots = data._modules[data._moduleID2Index(i)]
            time = 0
            for s in foundSlots:
                slots.append(s)
                for t in s:
                    if t["act"] !=Activity.prefetch:
                        time += t["finish"]-t["start"]
            modules[-1]['time']=time
        for i,m in parser._esModuleNames.items():
            modules.append({"name": f"{m}", "slots":[]})
            slots = modules[-1]["slots"]
            foundSlots = data._modules[data._moduleID2Index(-1*i)]
            time = 0
            for s in foundSlots:
                slots.append(s)
                for t in s:
                    if t["act"] !=Activity.prefetch:
                        time += t["finish"]-t["start"]
            modules[-1]['time']=time
        modules.sort(key= lambda x : x['time'], reverse=True)
        final['transitions'].append({"name": "source", "slots":sourceSlot})
        for m in modules:
            final['transitions'].append(m)

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
import unittest

class DummyFile(list):
    def __init__(self):
        super()
    def seek(self, i):
        pass

class TestModuleCommand(unittest.TestCase):
    def setUp(self):
        self.tracerFile = DummyFile()
        t = [0]
        def incr(t):
            t[0] += 1
            return t[0]
        
        self.tracerFile.extend([
            '#R 1 Record',
            '#M 1 Module',
            '#N 1 ESModule',
             f'F {Phase.startTracing} 0 0 0 0 {incr(t)}',
             f'S {Phase.construction} 0 {incr(t)}',
             f's {Phase.construction} 0 {incr(t)}3',
             f'M {Phase.construction} 0 1 0 0 0 {incr(t)}',
             f'm {Phase.construction} 0 1 0 0 0 {incr(t)}',
             f'F {Phase.beginJob} 0 0 0 0 {incr(t)}',
             f'M {Phase.beginJob} 0 1 0 0 0 {incr(t)}',
             f'm {Phase.beginJob} 0 1 0 0 0 {incr(t)}',
             f'f {Phase.beginJob} 0 0 0 0 {incr(t)}',
             f'F {Phase.beginProcessBlock} 0 0 0 0 {incr(t)}',
             f'f {Phase.beginProcessBlock} 0 0 0 0 {incr(t)}',
             f'S {Phase.getNextTransition} {incr(t)}',
             f's {Phase.getNextTransition} {incr(t)}',
             f'F {Phase.esSyncEnqueue} -1 1 0 0 {incr(t)}',
             f'F {Phase.esSync} -1 1 0 0 {incr(t)}',
             f'f {Phase.esSync} -1 1 0 0 {incr(t)}',
             f'S {Phase.globalBeginRun} 0 {incr(t)}',
             f's {Phase.globalBeginRun} 0 {incr(t)}',
             f'S {Phase.getNextTransition} {incr(t)}',
             f's {Phase.getNextTransition} {incr(t)}',
             f'F {Phase.globalBeginRun} 0 1 0 0 {incr(t)}',
             f'P {Phase.globalBeginRun} 0 1 0 0 0 {incr(t)}',
             f'p {Phase.globalBeginRun} 0 1 0 0 0 {incr(t)}',
             f'M {Phase.globalBeginRun} 0 1 0 0 0 {incr(t)}',
             f'm {Phase.globalBeginRun} 0 1 0 0 0 {incr(t)}',
             f'f {Phase.globalBeginRun} 0 1 0 0 {incr(t)}',
             f'F {Phase.esSyncEnqueue} -1 1 1 0 {incr(t)}',
             f'F {Phase.esSync} -1 1 1 0 {incr(t)}',
             f'f {Phase.esSync} -1 1 1 0 {incr(t)}',
             f'S {Phase.getNextTransition} {incr(t)}',
             f's {Phase.getNextTransition} {incr(t)}',
             f'F {Phase.streamBeginRun} 0 1 0 0 {incr(t)}',
             f'M {Phase.streamBeginRun} 0 1 0 0 0 {incr(t)}',
             f'm {Phase.streamBeginRun} 0 1 0 0 0 {incr(t)}',
             f'f {Phase.streamBeginRun} 0 1 0 0 {incr(t)}',
             f'F {Phase.streamBeginRun} 1 1 0 0 {incr(t)}',
             f'M {Phase.streamBeginRun} 1 1 0 0 0 {incr(t)}',
             f'm {Phase.streamBeginRun} 1 1 0 0 0 {incr(t)}',
             f'f {Phase.streamBeginRun} 1 1 0 0 {incr(t)}',
             f'S {Phase.globalBeginLumi} 0 {incr(t)}',
             f's {Phase.globalBeginLumi} 0 {incr(t)}',
             f'S {Phase.getNextTransition} {incr(t)}',
             f's {Phase.getNextTransition} {incr(t)}',
             f'F {Phase.globalBeginLumi} 0 1 1 0 {incr(t)}',
             f'P {Phase.globalBeginLumi} 0 1 0 0 0 {incr(t)}',
             f'p {Phase.globalBeginLumi} 0 1 0 0 0 {incr(t)}',
             f'M {Phase.globalBeginLumi} 0 1 0 0 0 {incr(t)}',
             f'm {Phase.globalBeginLumi} 0 1 0 0 0 {incr(t)}',
             f'f {Phase.globalBeginLumi} 0 1 1 0 {incr(t)}',
             f'F {Phase.streamBeginLumi} 0 1 1 0 {incr(t)}',
             f'f {Phase.streamBeginLumi} 0 1 1 0 {incr(t)}',
             f'F {Phase.streamBeginLumi} 1 1 1 0 {incr(t)}',
             f'f {Phase.streamBeginLumi} 1 1 1 0 {incr(t)}',
             f'S {Phase.Event} 0 {incr(t)}',
             f's {Phase.Event} 0 {incr(t)}',
             f'S {Phase.getNextTransition} {incr(t)}',
             f's {Phase.getNextTransition} {incr(t)}',
             f'F {Phase.Event} 0 1 1 1 {incr(t)}',
             f'S {Phase.Event} 1 {incr(t)}',
             f's {Phase.Event} 1 {incr(t)}',
             f'F {Phase.Event} 1 1 1 2 {incr(t)}',
             f'P {Phase.Event} 0 1 0 0 0 {incr(t)}',
             f'p {Phase.Event} 0 1 0 0 0 {incr(t)}',
             f'Q {Phase.Event} 0 1 1 0 1 0 {incr(t)}',
             f'q {Phase.Event} 0 1 1 0 1 0 {incr(t)}',
             f'N {Phase.Event} 0 1 1 0 1 0 {incr(t)}',
             f'n {Phase.Event} 0 1 1 0 1 0 {incr(t)}',
             f'P {Phase.Event} 1 1 0 0 0 {incr(t)}',
             f'p {Phase.Event} 1 1 0 0 0 {incr(t)}',
             f'M {Phase.Event} 0 1 0 0 0 {incr(t)}',
             f'M {Phase.Event} 1 1 0 0 0 {incr(t)}',
             f'm {Phase.Event} 1 1 0 0 0 {incr(t)}',
             f'm {Phase.Event} 0 1 0 0 0 {incr(t)}',
             f'f {Phase.Event} 0 1 1 1 {incr(t)}',
             f'f {Phase.Event} 1 1 1 2 {incr(t)}'])

        None
    def testContainers(self):
        c = Containers()
        c.indexedGlobal(2)
        self.assertEqual(len(c.allGlobals()), 3)
        c.indexedStream(2)
        self.assertEqual(len(c.allStreams()), 3)
        slot = c.findOpenSlotInModGlobals(2, 1)
        self.assertEqual(len(c._modGlobals),3)
        self.assertEqual(len(slot),0)
        slot.append({"start":1, "finish":0, "id":1})
        def testFind(item):
            return item["id"]==1
        item,s = c.findLastInModGlobals(2, 1, testFind)
        self.assertEqual(item["id"],1)
        self.assertEqual(slot,s)
        slot = c.findOpenSlotInModStreams(2, 1)
        self.assertEqual(len(c._modStreams),3)
        self.assertEqual(len(slot),0)
        slot.append({"start":1, "finish":0, "id":1})
        item,s = c.findLastInModStreams(2, 1, testFind)
        self.assertEqual(item["id"],1)
        self.assertEqual(slot,s)
    def testFrameworkOnly(self):
        parser = TracerCompactFileParser(self.tracerFile, True, False)
        j = jsonInfo(parser)
        #print(j)
        self.assertEqual(len(j["modules"]), 0)
        self.assertEqual(len(j["esModules"]), 0)
        self.assertEqual(len(j['transitions']), 3)
        self.assertEqual(j['transitions'][0]['name'], "Global")
        self.assertEqual(j['transitions'][1]['name'], "Stream 0")
        self.assertEqual(j['transitions'][2]['name'], "Stream 1")
        self.assertEqual(len(j["transitions"][0]["slots"]), 1)
        self.assertEqual(len(j["transitions"][0]["slots"][0]), 15)
        self.assertEqual(len(j["transitions"][1]["slots"]), 1)
        self.assertEqual(len(j["transitions"][1]["slots"][0]), 5)
        self.assertEqual(len(j["transitions"][2]["slots"]), 1)
        self.assertEqual(len(j["transitions"][2]["slots"][0]), 5)
    def testFull(self):
        parser = TracerCompactFileParser(self.tracerFile, False, False)
        j = jsonInfo(parser)
        #print(j)
        self.assertEqual(len(j["modules"]), 2)
        self.assertEqual(len(j["esModules"]), 2)
        self.assertEqual(len(j['transitions']), 3)
        self.assertEqual(j['transitions'][0]['name'], "Global")
        self.assertEqual(j['transitions'][1]['name'], "Stream 0")
        self.assertEqual(j['transitions'][2]['name'], "Stream 1")
        self.assertEqual(len(j["transitions"][0]["slots"]), 2)
        self.assertEqual(len(j["transitions"][0]["slots"][0]), 15)
        self.assertEqual(len(j["transitions"][0]["slots"][1]), 6)
        self.assertEqual(len(j["transitions"][1]["slots"]), 2)
        self.assertEqual(len(j["transitions"][1]["slots"][0]), 5)
        self.assertEqual(len(j["transitions"][1]["slots"][1]), 5)
        self.assertEqual(len(j["transitions"][2]["slots"]), 2)
        self.assertEqual(len(j["transitions"][2]["slots"][0]), 5)
        self.assertEqual(len(j["transitions"][2]["slots"][1]), 3)
    def testModuleCentric(self):
        parser = TracerCompactFileParser(self.tracerFile, False, True)
        j = jsonInfo(parser)
        #print(j)
        self.assertEqual(len(j["modules"]), 2)
        self.assertEqual(len(j["esModules"]), 2)
        self.assertEqual(len(j['transitions']), 6)
        self.assertEqual(j['transitions'][0]['name'], "Global")
        self.assertEqual(j['transitions'][1]['name'], "Stream 0")
        self.assertEqual(j['transitions'][2]['name'], "Stream 1")
        self.assertEqual(j['transitions'][3]['name'], "source")
        self.assertEqual(j['transitions'][4]['name'], "Module")
        self.assertEqual(j['transitions'][5]['name'], "ESModule")
        self.assertEqual(len(j["transitions"][0]["slots"]), 1)
        self.assertEqual(len(j["transitions"][0]["slots"][0]), 15)
        self.assertEqual(len(j["transitions"][1]["slots"]), 1)
        self.assertEqual(len(j["transitions"][1]["slots"][0]), 5)
        self.assertEqual(len(j["transitions"][2]["slots"]), 1)
        self.assertEqual(len(j["transitions"][2]["slots"][0]), 5)
        self.assertEqual(len(j["transitions"][4]["slots"]), 2)
        self.assertEqual(len(j["transitions"][4]["slots"][0]), 10)
        self.assertEqual(len(j["transitions"][4]["slots"][1]), 2)
        self.assertTrue(j["transitions"][4]["slots"][1][-1]['finish'] != 0.0)
        self.assertEqual(len(j["transitions"][5]["slots"]), 1)
        self.assertEqual(len(j["transitions"][5]["slots"][0]), 2)

def runTests():
    return unittest.main(argv=sys.argv[:1])

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
    parser.add_argument('-m', '--module_centric',
                        action = 'store_true',
                        help='''For --json or --web, organize data by module instead of by global/stream.''' )
    parser.add_argument('-t', '--test',
                        action='store_true',
                        help='''Run internal tests.''')

    args = parser.parse_args()
    if args.test:
        runTests()
    else :
        parser = TracerCompactFileParser(args.filename, args.frameworkOnly, args.module_centric)
        if args.json or args.web:
            j = json.dumps(jsonInfo(parser))
            if args.json:
                print(j)
            if args.web:
                f=open('data.json', 'w')
                f.write(j)
                f.close()
        else:
            textOutput(parser)
