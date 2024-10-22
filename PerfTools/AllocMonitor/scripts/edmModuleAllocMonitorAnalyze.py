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
To Use: Add the ModuleAllocMonitor Service to the cmsRun job use something like this
  in the configuration:

  process.add_(cms.Service("ModuleAllocMonitor", fileName = cms.untracked.string("moduleAlloc.log")))

  After running the job, execute this script and pass the name of the
  ModuleAllocMonitor log file to the script.

  This script will output a more human readable form of the data in the log file.'''
    return s

#these values come from moduleALloc_setupFile.cc
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

activityNames_ = { Activity.prefetch : 'prefetch',
                   Activity.acquire : 'acquire',
                   Activity.process : 'process',
                   Activity.delayedGet : 'delayedGet',
                   Activity.externalWork : 'externalWork' }

def activityName(activity):
    return activityNames_[activity]
  
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
    Phase.getNextTransition: 'get next transition',
    Phase.openFile : "open file"
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
    Phase.getNextTransition,
    Phase.openFile
}
def transitionIsGlobal(transition):
    return transition in globalTransitions_;

def textPrefix_(time, indentLevel):
    #using 11 spaces for time should accomodate a job that runs 24 hrs
    return f'{time:>11} '+"++"*indentLevel

class AllocInfo(object):
    def __init__(self,payload):
        self.nAllocs = int(payload[0])
        self.nDeallocs = int(payload[1])
        self.added = int(payload[2])
        self.minTemp = int(payload[3])
        self.maxTemp = int(payload[4])
        self.max1Alloc = int(payload[5])
    def inject(self, transition):
        transition["nAllocs"]=self.nAllocs
        transition["nDeallocs"]=self.nDeallocs
        transition["added"]=self.added
        transition["minTemp"]=self.minTemp
        transition["maxTemp"]=self.maxTemp
        transition["max1Alloc"]=self.max1Alloc
    def __repr__(self):
        return "{{'nAlloc': {}, 'nDealloc': {}, 'added': {}, 'minTemp': {}, 'maxTemp': {}, 'max1Alloc': {} }}".format(self.nAllocs, self.nDeallocs, self.added, self.minTemp, self.maxTemp, self.max1Alloc)
    def toSimpleDict(self):
        return {'nAlloc' : self.nAllocs, 'nDealloc' :self.nDeallocs, 'added' : self.added, 'minTemp' : self.minTemp, 'maxTemp' : self.maxTemp, 'max1Alloc' : self.max1Alloc }
        
class SyncValues(object):
    def __init__(self):
        self._runs = []
        self._lumis = []
        self._streams = []
    def setRun(self, index, runNumber):
        while len(self._runs) <= index:
            self._runs.append(0)
        self._runs[index] = runNumber
    def runFor(self,index):
        return self._runs[index]
    def setLumi(self, index, runNumber, lumiNumber):
        while len(self._lumis) <= index:
            self._lumis.append((0,0))
        self._lumis[index] = (runNumber, lumiNumber)
    def lumiFor(self, index):
        return self._lumis[index]
    def setStream(self, index, runNumber, lumiNumber, eventNumber):
        while len(self._streams) <= index:
            self._streams.append((0,0,0))
        self._streams[index] = (runNumber, lumiNumber, eventNumber)
    def streamFor(self, index):
        return self._streams[index]
    def get(self, transition, index):
        if transition == Phase.construction or transition == Phase.destruction:
            return ()
        if transition == Phase.beginJob or transition == Phase.endJob or transition == Phase.openFile:
            return ()
        if transition == Phase.globalBeginRun or transition == Phase.globalEndRun or transition == Phase.globalWriteRun:
            return (self.runFor(index),)
        if transition == Phase.globalBeginLumi or transition == Phase.globalEndLumi or transition == Phase.globalWriteLumi:
            return self.lumiFor(index)
        if transition == Phase.getNextTransition:
            return ()
        if transition == Phase.writeProcessBlock:
            return ()
        if transition == Phase.beginStream:
            self.setStream(index, 0,0,0)
            return ()
        if not transitionIsGlobal(transition):
            return self.streamFor(index)
        raise RuntimeError("Unknown transition {}".format(transition))

class TempModuleTransitionInfos(object):
    def __init__(self):
        self._times = {}
        self._esTimes = {}
    def insertTime(self, label, transition, index, time):
        self._times[(label, transition, index)] = time
    def findTime(self, label, transition, index):
        time = self._times[(label, transition, index)]
        del self._times[(label, transition, index)]
        return time
    def insertTimeES(self, label, transition, index, record, call, time):
        self._esTimes[(label, transition, index, record, call)] = time
    def findTimeES(self, label, transition, index, record, call):
        time = self._esTimes[(label, transition, index, record, call)]
        del self._esTimes[(label, transition, index, record, call)]
        return time

class ModuleData(object):
    def __init__(self, start, stop, transition, sync, activity, allocInfo, recordName=None, callID=None):
        self.timeRange = (start, stop)
        self.transition = transition
        self.sync = sync
        self.activity = activity
        self.allocInfo = allocInfo
        self.record = (recordName, callID)
    def __repr__(self):
        if self.record[0]:
            return "{{ 'timeRange': {}, 'transition': {}, 'sync' :{}, 'activity':{}, 'record': {{'name' : {}, 'callID' :{} }}, 'alloc':{} }}".format(self.timeRange, self.transition, self.sync, self.activity, self.record[0], self.record[1], self.allocInfo)

        return "{{ 'timeRange': {}, 'transition': {}, 'sync' :{}, 'activity':{}, 'alloc':{} }}".format(self.timeRange, self.transition, self.sync, self.activity, self.allocInfo)
    def syncToSimpleDict(self):
        if len(self.sync) == 0:
            return self.sync
        if len(self.sync) == 1:
            return {'run' : self.sync[0]}
        if len(self.sync) == 2:
            return {'run' : self.sync[0], 'lumi' : self.sync[1] }
        return {'run' : self.sync[0], 'lumi' : self.sync[1], 'event' : self.sync[2] }
    def toSimpleDict(self) :
        if self.record[0]:
            return {'timeRange': self.timeRange, 'transition': transitionName(self.transition), 'sync' : self.syncToSimpleDict(), 'activity' : activityName(self.activity), 'record' :{'name': self.record[0], 'callID' : self.record[1] }, 'alloc' : self.allocInfo.toSimpleDict() }
        return {'timeRange': self.timeRange, 'transition': transitionName(self.transition), 'sync' : self.syncToSimpleDict(), 'activity': activityName(self.activity), 'alloc' : self.allocInfo.toSimpleDict() }
        
    
class ModuleCentricModuleData(object):
    def __init__(self):
        self._data = {}
        self._last = {}
        self._startTime = None
    def setStartTime(self, time):
        self._startTime = time
    def insert(self, label, start, stop, transition, index, sync, activity, allocInfo, recordName=None, callID=None):
        if label not in self._data:
            self._data[label] = []
        self._data[label].append(ModuleData(start, stop, transition, sync, activity, allocInfo, recordName, callID))
        self._last[(label, transition, index, activity)] = self._data[label][-1]
    def findLast(self, label, transition, index, activity):
        return self._last[(label, transition, index, activity)] 
    def __repr__(self):
        return str(self._data)
    def data(self):
        return self._data
    def toSimpleDict(self):
        dct = {'startedMonitoring': self._startTime, 'source' :[], 'clearEvent': [], 'modules' :{}}
        modules = dct['modules']
        for m,lst in self._data.items():
            l = None
            if m == 'source':
                l = dct['source']
            elif m == 'clearEvent':
                l = dct['clearEvent']
            else:
                modules[m]=[]
                l = modules[m]
            for d in lst:
                l.append( d.toSimpleDict() )
        return dct
    def sortModulesBy(self, attribute):
        data = []
        for m, lst in self._data.items():
            data.append((m, max(lst, key = lambda x: getattr(x.allocInfo,attribute))) )
        data.sort( key = lambda x: getattr(x[1].allocInfo,attribute), reverse=True)
        return list(map(lambda x: (x[0], x[1].toSimpleDict()), data))
    
class TemporalModuleData(object):
    def __init__(self):
        self._data = []
        self._last = {}
        self._startTime = None
    def setStartTime(self, time):
        self._startTime = time
    def insert(self, label, start, stop, transition, index, sync, activity, allocInfo, recordName=None, callID=None):
        self._data.append((label, ModuleData(start, stop, transition, sync, activity, allocInfo, recordName, callID)))
        self._last[(label,transition, index, activity)] = self._data[-1]
    def findLast(self, label, transition,index, activity):
        v = self._last.get((label, transition, index, activity), None)
        if v:
            return v[1]
        return None
    def __repr__(self):
        return str(self._data)
    def data(self):
        return self._data
    def toSimpleDict(self):
        dct = {'startedMonitoring': self._startTime, 'measurements' :[]}
        measurements = dct['measurements']
        for d in self._data:
            entry = d[1].toSimpleDict()
            entry['label'] = d[0]
            measurements.append(entry)
        return dct



    
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
    def jsonInfo(self, syncs, temp, data):
        isSourceTrans = False
        if self.transition == Phase.startTracing:
            data.setStartTime(self.time)
        elif self.transition == Phase.globalBeginRun:
            syncs.setRun(self.index, self.sync[0])
            isSourceTrans = True
        elif self.transition == Phase.globalBeginLumi:
            syncs.setLumi(self.index, self.sync[0], self.sync[1])
            isSourceTrans = True
        elif self.transition == Phase.Event:
            syncs.setStream(self.index, self.sync[0], self.sync[1], self.sync[2])
            isSourceTrans = True
        elif self.transition == Phase.clearEvent:
            temp.insertTime("clearEvent", self.transition, self.index, self.time)
        elif not transitionIsGlobal(self.index):
            syncs.setStream(self.index, self.sync[0], self.sync[1], self.sync[2])
        if isSourceTrans:
            src = data.findLast("source", self.transition, self.index, Activity.process)
            if src.sync != self.index:
                raise RuntimeError("Framework and Source transitions do not have matching index: source {} framework {} for transition type {} at framework time {} and source time {}".format(src.sync, self.index, self.transition, self.time, src.timeRange))
            src.sync = syncs.get(self.transition, self.index)
    def jsonVisInfo(self,  data):
        if transitionIsGlobal(self.transition):
            index = 0
            if self.transition == Phase.startTracing:
                data.indexedGlobal(0).append(jsonTransition(type=self.transition, id=index, sync=list(self.sync),start=0, finish=self.time ))
                return
            elif self.transition==Phase.globalBeginRun:
                index = self.index
                container = data.indexedGlobal(index)
                #find source, should be previous
                last = container[-1]
                if last["type"]==Phase.globalBeginRun and last["isSrc"]:
                    last["sync"]=list(self.sync)
            elif self.transition==Phase.globalBeginLumi:
                index = self.index
                container = data.indexedGlobal(index)
                #find source, should be previous
                last = container[-1]
                if last["type"]==Phase.globalBeginLumi and last["isSrc"]:
                    last["sync"]=list(self.sync)
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
        if self.transition == Phase.clearEvent:
            self.alloc = AllocInfo(payload[6:])
    def textSpecial(self):
        return "finished"
    def jsonInfo(self, syncs, temp, data):
        if self.transition == Phase.clearEvent:
            start = temp.findTime("clearEvent", self.transition, self.index)
            data.insert( "clearEvent" , start, self.time, self.transition, self.index, syncs.get(Phase.Event, self.index) , Activity.process, self.alloc)
    def jsonVisInfo(self,  data):
        if transitionIsGlobal(self.transition):
            index = findMatchingTransition(list(self.sync), data.allGlobals())
            container = data.indexedGlobal(index)
        else:
            container = data.indexedStream(self.index)
        container[-1]["finish"]=self.time*kMicroToSec
        if self.transition == Phase.clearEvent:
            self.alloc.inject(container[-1])



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
        if self.transition == Phase.openFile:
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
    def jsonInfo(self, syncs, temp, data):
        temp.insertTime("source", self.transition, self.index, self.time)
    def jsonVisInfo(self,  data):
        if self.transition == Phase.getNextTransition:
            data._nextTrans.append(jsonTransition(type=self.transition, id=self.index, sync=[0,0,0], start=self.time, finish=0, isSrc=True))
            if self._moduleCentric:
                #this all goes to a module ID sorted container so not knowing actual index is OK
                data.findOpenSlotInModGlobals(0,0).append(data._nextTrans[-1])
            return
        elif self.transition == Phase.construction:
            index = 0
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
        #print(payload)
        if self.index == -1:
            self.allocInfo = AllocInfo(payload[2:])
        else:
            self.allocInfo = AllocInfo(payload[3:])
        self._moduleCentric = moduleCentric
    def textSpecial(self):
        return "finished"
    def jsonInfo(self, syncs, temp, data):
        start = temp.findTime("source", self.transition, self.index)
        #we do not know the sync yet so have to wait until the framework transition
        if self.transition in [ Phase.construction, Phase.getNextTransition, Phase.destruction, Phase.openFile]:
            data.insert( "source" , start, self.time, self.transition, self.index, (0,) , Activity.process, self.allocInfo)
        else:
            data.insert( "source" , start, self.time, self.transition, self.index, self.index , Activity.process, self.allocInfo)
    def jsonVisInfo(self,  data):
        index = self.index
        if self.transition == Phase.Event:
            container = data.indexedStream(index)
        elif self.transition == Phase.getNextTransition:
            data._nextTrans[-1]['finish'] = self.time*kMicroToSec
            self.allocInfo.inject(data._nextTrans[-1])
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
                    self.allocInfo.inject(pre)
                    break
            return
        else:
            container = data.indexedGlobal(index)

        container[-1]["finish"]=self.time*kMicroToSec
        self.allocInfo.inject(container[-1])

class EDModuleTransitionParser(object):
    def __init__(self, payload, moduleNames):
        self.transition = int(payload[0])
        self.index = int(payload[1])
        self.moduleID = int(payload[2])
        self.moduleName = moduleNames[self.moduleID]
        self.callID = int(payload[3])
        self.time = int(payload[4])
    def baseIndentLevel(self):
        return transitionIndentLevel(self.transition)
    def textPrefix(self, context):
        indent = 0
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
    def _preJsonVis(self, activity,  data, mayUseTemp = False):
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
    def _postJsonVis(self,  data, alloc, injectAfter = None):
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
            alloc.inject(item)
            if injectAfter:
                slot.append(injectAfter)
    def _preJsonInfo(self, temp):
        temp.insertTime(self.moduleName, self.transition, self.index, self.time)
    def _postJsonInfo(self, syncs, temp, data, activity):
        start = temp.findTime(self.moduleName, self.transition, self.index)
        data.insert( self.moduleName , start, self.time, self.transition, self.index, syncs.get(self.transition, self.index) , activity, self.allocInfo)

                
class PreEDModuleTransitionParser(EDModuleTransitionParser):
    def __init__(self, payload, names, moduleCentric):
        super().__init__(payload, names)
        self._moduleCentric = moduleCentric
    def textSpecial(self):
        return "starting action"
    def jsonVisInfo(self,  data):
        return self._preJsonVis(Activity.process, data, mayUseTemp=self._moduleCentric)
    def jsonInfo(self, syncs, temp, data):
        self._preJsonInfo(temp)

    
class PostEDModuleTransitionParser(EDModuleTransitionParser):
    def __init__(self, payload, names):
        super().__init__(payload, names)
        self.allocInfo = AllocInfo(payload[5:])
    def textSpecial(self):
        return "finished action"
    def jsonInfo(self, syncs, temp, data):
        self._postJsonInfo(syncs, temp, data, Activity.process)
    def jsonVisInfo(self,  data):
        return self._postJsonVis(data, self.allocInfo)
        

class PreEDModuleAcquireParser(EDModuleTransitionParser):
    def __init__(self, payload, names, moduleCentric):
        super().__init__(payload, names)
        self._moduleCentric = moduleCentric
    def textSpecial(self):
        return "starting acquire"
    def jsonVisInfo(self,  data):
        return self._preJsonVis(Activity.acquire, data, mayUseTemp=self._moduleCentric)
    def jsonInfo(self, syncs, temp, data):
        self._preJsonInfo(temp)


class PostEDModuleAcquireParser(EDModuleTransitionParser):
    def __init__(self, payload, names, moduleCentric):
        super().__init__(payload, names)
        self.allocInfo = AllocInfo(payload[5:])
        self._moduleCentric = moduleCentric
    def textSpecial(self):
        return "finished acquire"
    def jsonVisInfo(self,  data):
        if self._moduleCentric:
            #inject an external work at end of the same slot to guarantee module run is in that slot
            return self._postJsonVis( data, jsonModuleTransition(type=self.transition, id=self.index, modID=self.moduleID, callID=self.callID, activity=Activity.externalWork, start=self.time))
        return self._postJsonVis(data, self.allocInfo)
    def jsonInfo(self, syncs, temp, data):
        self._postJsonInfo(syncs, temp, data, Activity.acquire)

class PreEDModuleEventDelayedGetParser(EDModuleTransitionParser):
    def __init__(self, payload, names):
        super().__init__(payload, names)
    def textSpecial(self):
        return "starting delayed get"
    def jsonVisInfo(self,  data):
        return self._preJsonVis(Activity.delayedGet, data)
    def jsonInfo(self, syncs, temp, data):
        pass
        #self._preJsonInfo(temp)

class PostEDModuleEventDelayedGetParser(EDModuleTransitionParser):
    def __init__(self, payload, names):
        super().__init__(payload, names)
        self.allocInfo = AllocInfo(payload[5:])
    def textSpecial(self):
        return "finished delayed get"
    def jsonVisInfo(self,  data):
        return self._postJsonVis(data, self.allocInfo)
    def jsonInfo(self, syncs, temp, data):
        pass
        #self._postJsonInfo(syncs, temp, data, Activity.delayedGet)

class PreEventReadFromSourceParser(EDModuleTransitionParser):
    def __init__(self, payload, names):
        super().__init__(payload, names)
    def textSpecial(self):
        return "starting read from source"
    def jsonVisInfo(self,  data):
        slot = self._preJsonVis(Activity.process, data)
        slot['isSrc'] = True
        return slot
    def jsonInfo(self, syncs, temp, data):
        temp.insertTime(self.moduleName+'source', self.transition, self.index, self.time)

class PostEventReadFromSourceParser(EDModuleTransitionParser):
    def __init__(self, payload, names):
        super().__init__(payload, names)
        self.allocInfo = AllocInfo(payload[5:])
    def textSpecial(self):
        return "finished read from source"
    def jsonVisInfo(self,  data):
        return self._postJsonVis(data, self.allocInfo)
    def jsonInfo(self, syncs, temp, data):
        start = temp.findTime(self.moduleName+'source', self.transition, self.index)
        data.insert( "source" , start, self.time, self.transition, self.index, syncs.get(self.transition, self.index) , Activity.delayedGet, self.allocInfo)

class ESModuleTransitionParser(object):
    def __init__(self, payload, moduleNames, esModuleNames, recordNames):
        self.transition = int(payload[0])
        self.index = int(payload[1])
        self.moduleID = int(payload[2])
        self.moduleName = esModuleNames[self.moduleID]
        self.recordID = int(payload[3])
        self.recordName = recordNames[self.recordID]
        self.callID = int(payload[4])
        self.time = int(payload[5])
    def baseIndentLevel(self):
        return transitionIndentLevel(self.transition)
    def textPrefix(self, context):
        indent = 0
        context[(self.transition, self.index, -1*self.moduleID, self.callID)] = indent+1
        return textPrefix_(self.time, indent+1+self.baseIndentLevel())
    def textPostfix(self):
        return f'esmodule {self.moduleName} in record {self.recordName} during {transitionName(self.transition)} : id={self.index}'
    def text(self, context):
        return f'{self.textPrefix(context)} {self.textSpecial()}: {self.textPostfix()}'
    def _preJsonVis(self, activity,  data):
        index = self.index
        if transitionIsGlobal(self.transition):
            slot = data.findOpenSlotInModGlobals(index, -1*self.moduleID)
        else:
            slot = data.findOpenSlotInModStreams(index, -1*self.moduleID)
        slot.append(jsonModuleTransition(type=self.transition, id=self.index, modID=-1*self.moduleID, callID=self.callID, activity=activity, start=self.time))
        return slot[-1]
    def _postJsonVis(self,  data, alloc):
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
        alloc.inject(item)
    def _preJsonInfo(self, temp):
        temp.insertTimeES(self.moduleName, self.transition, self.index, self.recordID, self.callID, self.time)
    def _postJsonInfo(self, syncs, temp, data, activity):
        start = temp.findTimeES(self.moduleName, self.transition, self.index, self.recordID, self.callID)
        data.insert( self.moduleName , start, self.time, self.transition, self.index, syncs.get(self.transition, self.index) , activity, self.allocInfo, self.recordName, self.callID)


class PreESModuleTransitionParser(ESModuleTransitionParser):
    def __init__(self, payload, names, esNames, recordNames):
        super().__init__(payload, names, esNames, recordNames)
    def textSpecial(self):
        return "starting action"
    def jsonVisInfo(self,  data):
        return self._preJsonVis(Activity.process, data)
    def jsonInfo(self, syncs, temp, data):
        self._preJsonInfo(temp)

class PostESModuleTransitionParser(ESModuleTransitionParser):
    def __init__(self, payload, names, esNames, recordNames):
        super().__init__(payload, names, esNames, recordNames)
        self.allocInfo = AllocInfo(payload[6:])
    def textSpecial(self):
        return "finished action"
    def jsonVisInfo(self,  data):
        return self._postJsonVis(data,self.allocInfo)
    def jsonInfo(self, syncs, temp, data):
        self._postJsonInfo(syncs, temp, data, Activity.process)

class PreESModuleAcquireParser(ESModuleTransitionParser):
    def __init__(self, payload, names, recordNames):
        super().__init__(payload, names, recordNames)
    def textSpecial(self):
        return "starting acquire"
    def jsonVisInfo(self,  data):
        return self._preJsonVis(Activity.acquire, data)

class PostESModuleAcquireParser(ESModuleTransitionParser):
    def __init__(self, payload, names, esNames, recordNames):
        super().__init__(payload, names, esNames, recordNames)
        self.allocInfo = AllocInfo(payload[6:])
    def textSpecial(self):
        return "finished acquire"
    def jsonVisInfo(self,  data):
        return self._postJsonVis(data, self.allocInfo)


def lineParserFactory (step, payload, moduleNames, esModuleNames, recordNames, moduleCentric):
    if step == 'F':
        parser = PreFrameworkTransitionParser(payload)
        return parser
    if step == 'f':
        return PostFrameworkTransitionParser(payload)
    if step == 'S':
        return PreSourceTransitionParser(payload, moduleCentric)
    if step == 's':
        return PostSourceTransitionParser(payload, moduleCentric)
    if step == 'M':
        return PreEDModuleTransitionParser(payload, moduleNames, moduleCentric)
    if step == 'm':
        return PostEDModuleTransitionParser(payload, moduleNames)
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
    if step == 'B':
        return PreESModuleAcquireParser(payload, moduleNames, esModuleNames, recordNames)
    if step == 'b':
        return PostESModuleAcquireParser(payload, moduleNames, esModuleNames, recordNames)
    raise LogicError("Unknown step '{}'".format(step))
    
#----------------------------------------------
def processingStepsFromFile(f,moduleNames, esModuleNames, recordNames, moduleCentric):
    for rawl in f:
        l = rawl.strip()
        if not l or l[0] == '#':
            continue
        (step,payload) = tuple(l.split(None,1))
        payload=payload.split()

        parser = lineParserFactory(step, payload, moduleNames, esModuleNames, recordNames, moduleCentric)
        if parser:
            yield parser
    return

class ModuleAllocCompactFileParser(object):
    def __init__(self,f, moduleCentric):
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
        return processingStepsFromFile(self._f,self._moduleNames, self._esModuleNames, self._recordNames,  self._moduleCentric)

def textOutput( parser ):
    context = {}
    for p in parser.processingSteps():
        print(p.text(context))
    
class VisualizationContainers(object):
    def __init__(self):
        self._modGlobals = [[]]
        self._modStreams = [[]]
        self._globals = [[]]
        self._streams = [[]]
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
    

class ModuleCentricVisualizationContainers(object):
    def __init__(self):
        self._modules= []
        self._globals = [[]]
        self._streams = [[]]
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


class ModuleCentricContainers(object):
    def __init__(self):
        self._modules= []
        self._globals = [[]]
        self._streams = [[]]
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

def jsonInfo(parser, temporal):
    sync = SyncValues()
    #used to keep track of outstanding module transitions
    temp = TempModuleTransitionInfos()
    if temporal:
        data = TemporalModuleData()
    else:
        data = ModuleCentricModuleData()
    for p in parser.processingSteps():
        if hasattr(p, "jsonInfo"):
            p.jsonInfo(sync,temp, data)
    return data

def sortByAttribute(parser, attribute):
    sync = SyncValues()
    #used to keep track of outstanding module transitions
    temp = TempModuleTransitionInfos()
    data = ModuleCentricModuleData()
    for p in parser.processingSteps():
        if hasattr(p, "jsonInfo"):
            p.jsonInfo(sync,temp, data)
    return data.sortModulesBy(attribute)
    
def jsonVisualizationInfo(parser):
    if parser._moduleCentric:
        data = ModuleCentricContainers()
    else:
        data = VisualizationContainers()
    for p in parser.processingSteps():
        p.jsonVisInfo( data)
    #make sure everything is sorted
    for g in data.allGlobals():
        g.sort(key=startTime)
    final = {"transitions" : [] , "modules": [], "esModules": []}
    final["transitions"].append({ "name":"Global", "slots": []})
    globals = final["transitions"][-1]["slots"]
    for i, g in enumerate(data.allGlobals()):
        globals.append(g)
        if not parser._moduleCentric:
            if len(data._modGlobals) < i+1:
                break
            for mod in data._modGlobals[i]:
                globals.append(mod)
    for i,s in enumerate(data.allStreams()):
        final["transitions"].append({"name": f"Stream {i}", "slots":[]})
        stream = final["transitions"][-1]["slots"]
        stream.append(s)
        if not parser._moduleCentric:
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
             f's {Phase.construction} 0 {incr(t)} 1 1 10 0 10 10',
             f'M {Phase.construction} 0 1 0 {incr(t)}',
             f'm {Phase.construction} 0 1 0 {incr(t)} 3 2 20 0 50 25',
             f'F {Phase.beginJob} 0 0 0 0 {incr(t)}',
             f'M {Phase.beginJob} 0 1 0 {incr(t)}',
             f'm {Phase.beginJob} 0 1 0 {incr(t)} 3 2 20 0 50 25',
             f'f {Phase.beginJob} 0 0 0 0 {incr(t)}',
             f'F {Phase.beginProcessBlock} 0 0 0 0 {incr(t)}',
             f'f {Phase.beginProcessBlock} 0 0 0 0 {incr(t)}',
             f'S {Phase.getNextTransition} {incr(t)}',
             f's {Phase.getNextTransition} {incr(t)} 1 1 10 0 10 10',
             f'S {Phase.globalBeginRun} 0 {incr(t)}',
             f's {Phase.globalBeginRun} 0 {incr(t)} 1 1 10 0 10 10',
             f'S {Phase.getNextTransition} {incr(t)}',
             f's {Phase.getNextTransition} {incr(t)} 1 1 10 0 10 10',
             f'F {Phase.globalBeginRun} 0 1 0 0 {incr(t)}',
             f'M {Phase.globalBeginRun} 0 1 0 {incr(t)}',
             f'm {Phase.globalBeginRun} 0 1 0 {incr(t)} 3 2 20 0 50 25',
             f'f {Phase.globalBeginRun} 0 1 0 0 {incr(t)}',
             f'S {Phase.getNextTransition} {incr(t)}',
             f's {Phase.getNextTransition} {incr(t)} 1 1 10 0 10 10',
             f'F {Phase.streamBeginRun} 0 1 0 0 {incr(t)}',
             f'M {Phase.streamBeginRun} 0 1 0 {incr(t)}',
             f'm {Phase.streamBeginRun} 0 1 0 {incr(t)} 3 2 20 0 50 25',
             f'f {Phase.streamBeginRun} 0 1 0 0 {incr(t)}',
             f'F {Phase.streamBeginRun} 1 1 0 0 {incr(t)}',
             f'M {Phase.streamBeginRun} 1 1 0 {incr(t)}',
             f'm {Phase.streamBeginRun} 1 1 0 {incr(t)} 3 2 20 0 50 25',
             f'f {Phase.streamBeginRun} 1 1 0 0 {incr(t)}',
             f'S {Phase.globalBeginLumi} 0 {incr(t)}',
             f's {Phase.globalBeginLumi} 0 {incr(t)} 1 1 10 0 10 10',
             f'S {Phase.getNextTransition} {incr(t)}',
             f's {Phase.getNextTransition} {incr(t)} 1 1 10 0 10 10',
             f'F {Phase.globalBeginLumi} 0 1 1 0 {incr(t)}',
             f'M {Phase.globalBeginLumi} 0 1 0 {incr(t)}',
             f'm {Phase.globalBeginLumi} 0 1 0 {incr(t)} 3 2 20 0 50 25',
             f'f {Phase.globalBeginLumi} 0 1 1 0 {incr(t)}',
             f'F {Phase.streamBeginLumi} 0 1 1 0 {incr(t)}',
             f'f {Phase.streamBeginLumi} 0 1 1 0 {incr(t)}',
             f'F {Phase.streamBeginLumi} 1 1 1 0 {incr(t)}',
             f'f {Phase.streamBeginLumi} 1 1 1 0 {incr(t)}',
             f'S {Phase.Event} 0 {incr(t)}',
             f's {Phase.Event} 0 {incr(t)} 1 1 10 0 10 10',
             f'S {Phase.getNextTransition} {incr(t)}',
             f's {Phase.getNextTransition} {incr(t)} 1 1 10 0 10 10',
             f'F {Phase.Event} 0 1 1 1 {incr(t)}',
             f'S {Phase.Event} 1 {incr(t)}',
             f's {Phase.Event} 1 {incr(t)} 1 1 10 0 10 10',
             f'F {Phase.Event} 1 1 1 2 {incr(t)}',
             f'N {Phase.Event} 0 1 1 0 {incr(t)}',
             f'n {Phase.Event} 0 1 1 0 {incr(t)} 6 5 30 0 100 80',
             f'M {Phase.Event} 0 1 0 {incr(t)}',
             f'M {Phase.Event} 1 1 0 {incr(t)}',
             f'm {Phase.Event} 1 1 0 {incr(t)} 3 2 20 0 50 25',
             f'm {Phase.Event} 0 1 0 {incr(t)} 3 2 20 0 50 25',
             f'f {Phase.Event} 0 1 1 1 {incr(t)}',
             f'f {Phase.Event} 1 1 1 2 {incr(t)}'])

        None
    def testSyncValues(self):
        s = SyncValues()
        s.setRun(0,1)
        self.assertEqual(s.runFor(0), 1)
        s.setLumi(0,1, 1)
        self.assertEqual(s.lumiFor(0), (1,1))
        s.setStream(1, 1,1,3)
        self.assertEqual(s.streamFor(1), (1,1,3))
    def testContainers(self):
        c = VisualizationContainers()
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
    def testJson(self):
        parser = ModuleAllocCompactFileParser(self.tracerFile, False)
        j = jsonInfo(parser, False)
        self.assertEqual(len(j.data()),3) 
        self.assertEqual(len(j.data()["source"]), 10)
        self.assertEqual(len(j.data()["Module"]), 8)
        self.assertEqual(len(j.data()["ESModule"]), 1)
    def testJsonTemporal(self):
        parser = ModuleAllocCompactFileParser(self.tracerFile, True)
        j = jsonInfo(parser, True)
        self.assertEqual(len(j.data()),19)
    def testSortBy(self):
        parser = ModuleAllocCompactFileParser(self.tracerFile, True)
        d = sortByAttribute(parser, 'maxTemp')
        #print(d)
        self.assertEqual(len(d), 3)
        self.assertEqual(d[0][0], 'ESModule')
        self.assertEqual(d[1][0], 'Module')
        self.assertEqual(d[2][0], 'source')
    def testFullVisualization(self):
        parser = ModuleAllocCompactFileParser(self.tracerFile, False)
        j = jsonVisualizationInfo(parser)
        #print(j)
        self.assertEqual(len(j["modules"]), 2)
        self.assertEqual(len(j["esModules"]), 2)
        self.assertEqual(len(j['transitions']), 3)
        self.assertEqual(j['transitions'][0]['name'], "Global")
        self.assertEqual(j['transitions'][1]['name'], "Stream 0")
        self.assertEqual(j['transitions'][2]['name'], "Stream 1")
        self.assertEqual(len(j["transitions"][0]["slots"]), 2)
        self.assertEqual(len(j["transitions"][0]["slots"][0]), 11)
        self.assertEqual(len(j["transitions"][0]["slots"][1]), 4)
        self.assertEqual(len(j["transitions"][1]["slots"]), 2)
        self.assertEqual(len(j["transitions"][1]["slots"][0]), 5)
        self.assertEqual(len(j["transitions"][1]["slots"][1]), 3)
        self.assertEqual(len(j["transitions"][2]["slots"]), 2)
        self.assertEqual(len(j["transitions"][2]["slots"][0]), 5)
        self.assertEqual(len(j["transitions"][2]["slots"][1]), 2)
    def testModuleCentricVisualization(self):
        parser = ModuleAllocCompactFileParser(self.tracerFile, True)
        j = jsonVisualizationInfo(parser)
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
        self.assertEqual(len(j["transitions"][0]["slots"][0]), 11)
        self.assertEqual(len(j["transitions"][1]["slots"]), 1)
        self.assertEqual(len(j["transitions"][1]["slots"][0]), 5)
        self.assertEqual(len(j["transitions"][2]["slots"]), 1)
        self.assertEqual(len(j["transitions"][2]["slots"][0]), 5)
        self.assertEqual(len(j["transitions"][4]["slots"]), 2)
        self.assertEqual(len(j["transitions"][4]["slots"][0]), 7)
        self.assertEqual(len(j["transitions"][4]["slots"][1]), 1)
        self.assertTrue(j["transitions"][4]["slots"][1][-1]['finish'] != 0.0)
        self.assertEqual(len(j["transitions"][5]["slots"]), 1)
        self.assertEqual(len(j["transitions"][5]["slots"][0]), 1)

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
    parser.add_argument('-j', '--json',
                        action='store_true',
                        help='''Write output in json format.''' )
    parser.add_argument('-s', '--sortBy',
                        default = '',
                        type = str,
                        help="sort modules by attribute. Alloed values 'nAllocs', 'nDeallocs', 'added', 'minTemp', maxTemp', and 'max1Alloc'")
#    parser.add_argument('-w', '--web',
#                        action='store_true',
#                        help='''Writes data.js file that can be used with the web based inspector. To use, copy directory ${CMSSW_RELEASE_BASE}/src/FWCore/Services/template/web to a web accessible area and move data.js into that directory.''')
    parser.add_argument('-t', '--timeOrdered',
                        action = 'store_true',
                        help='''For --json, organize data by time instead of by module.''' )
    parser.add_argument('-T', '--test',
                        action='store_true',
                        help='''Run internal tests.''')

    args = parser.parse_args()
    if args.test:
        runTests()
    else :
        parser = ModuleAllocCompactFileParser(args.filename, not args.timeOrdered)
#        if args.json or args.web:
        if args.json:
            json.dump(jsonInfo(parser, args.timeOrdered).toSimpleDict(), sys.stdout, indent=2)
#           if args.web:
#                j ='export const data = ' + j
#                f=open('data.js', 'w')
#                f.write(j)
#                f.close()
        elif args.sortBy:
            print(json.dumps(sortByAttribute(parser, args.sortBy), indent=2))
        else:
            textOutput(parser)
