#!/usr/bin/env python3
class ModuleInfo(object):
    def __init__(self, label, type_):
        self._label = label
        self._type = type_
        self._streamInfo = dict()
        self._eventInfo = list()
    def __repr__(self):
        return self._label+" "+self._type+" "+str(self._eventInfo)
class ModuleCall(object):
    def __init__(self, data):
        self._temp = data[0]
        self._nTemp = data[1]
        self._unmatched = data[2]
        self._nUnmatched = data[3]
        self._new = data[4]
        self._nNew = data[5]

class ModuleEventInfo(object):
    def __init__(self, modCall, data, streamID):
        self._temp = modCall._temp
        self._nTemp = modCall._nTemp
        self._unmatched = modCall._unmatched
        self._nUnmatched = modCall._nUnmatched
        self._dataProdAlloc = data[0]
        self._nDataProdAlloc = data[1]
        self._new = modCall._new - self._dataProdAlloc
        self._nNew = modCall._nNew - self._nDataProdAlloc
        self._streamID = streamID
    def __repr__(self):
        return "temp("+str(self._temp)+","+str(self._nTemp)+") un("+str(self._unmatched)+","+str(self._nUnmatched)+") prod("+str(self._dataProdAlloc)+","+str(self._nDataProdAlloc)+") new("+str(self._new)+","+str(self._nNew)+")"
class FileParser(object):
    def __init__(self):
        self.modules = dict()
        self.acquires = dict()
        self.nStreams = 0
        pass
    def parse(self,file):
        for l in file:
            self._parseLine(l[:-1])
    def _parseLine(self, line):
        if len(line) == 0:
            return
        if line[0] == '#':
            return
        if line[0] == '@':
            self.addModule(line[2:])
        if line[0] == 'M':
            self.moduleCall(line[2:])
        if line[0] == 'A':
            self.moduleAcquireCall(line[2:])
        if line[0] == 'D':
            self.eventDeallocCall(line[2:])
    def addModule(self, l):
        d = l.split(" ")
        name, type_, index = d[0:3]
        self.modules[int(index)] = ModuleInfo(name,type_)
    def moduleCall(self, l):
        d = [int(x) for x in l.split(" ")]
        m = ModuleCall(d[2:])
        moduleName, streamID = d[0:2]
        if streamID+1 > self.nStreams:
            self.nStreams = streamID + 1
        self.modules[moduleName]._streamInfo[streamID] = m
        if d[0] in self.acquires:
            a = self.acquires[moduleName][streamID]
            m._temp += a._temp
            m._nTemp += a._nTemp
            m._unmatched += a._unmatched
            m._nUnmatched += a._nUnmatched
    def moduleAcquireCall(self,l):
        d = [int(x) for x in l.split(" ")]
        moduleName, streamID = d[0:2]
        m = self.modules[moduleName]
        self.acquires.setdefault(moduleName,dict())[streamID] = ModuleCall(d[2:])
        pass
    def eventDeallocCall(self,l):
        d = [int(x) for x in l.split(" ")]
        moduleName, streamID = d[0:2]
        streamInfo = self.modules[moduleName]._streamInfo[streamID]
        del self.modules[moduleName]._streamInfo[streamID]
        self.modules[moduleName]._eventInfo.append(ModuleEventInfo(streamInfo,d[2:],streamID))

def reportModulesWithMemoryGrowth(fileParser, showEvents):
    ret = []
    for m in fileParser.modules.values():
        mem = 0
        if len(m._eventInfo):
            l = list()
            previousNewInStream = [0]*fileParser.nStreams
            #skip first event as they often have initialization memory
            for e in m._eventInfo:
                l.append((previousNewInStream[e._streamID], e._unmatched))
                mem += previousNewInStream[e._streamID] - e._unmatched
                previousNewInStream[e._streamID] = e._new
            if mem and mem > m._eventInfo[0]._new:
                increment = []
                for n,u in l:
                    if len(increment):
                        increment[-1] -= u
                    increment.append(n)
                if showEvents:
                    ret.append((m._label,m._type, mem,increment))
                else:
                    ret.append((m._label, m._type, mem))
    return ret

def reportModuleDataProductMemory(fileParser, showEvents):
    ret = []
    for m in fileParser.modules.values():
        l = list()
        retained = False
        sum = 0
        for e in m._eventInfo:
            l.append(e._dataProdAlloc)
            if e._dataProdAlloc > 0:
                sum += e._dataProdAlloc
                retained = True
        if retained:
            if showEvents:
                ret.append((m._label, m._type, float(sum)/len(l), l))
            else:
                ret.append((m._label, m._type, float(sum)/len(l)))
    return ret
                
def reportModuleRetainingMemory(fileParser, showEvents):
    ret =[]
    for m in fileParser.modules.values():
        l = list()
        retained = False
        sum = 0
        for e in m._eventInfo[1:]:
            l.append(e._new)
            if e._new > 0:
                sum += e._new
                retained = True
        if retained:
            if showEvents:
                ret.append((m._label, m._type, float(sum)/len(l), l))
            else:
                ret.append((m._label, m._type, float(sum)/len(l)))
    return ret

def reportModuleTemporary(fileParser, showEvents):
    ret = []
    for m in fileParser.modules.values():
        l = list()
        retained = False
        sum = 0
        for e in m._eventInfo:
            l.append(e._temp)
            if e._temp > 0:
                sum += e._temp
                retained = True
        if retained:
            if showEvents:
                ret.append((m._label, m._type, float(sum)/len(l), l))
            else:
                ret.append((m._label, m._type, float(sum)/len(l)))
    return ret

def reportModuleNTemporary(fileParser, showEvents):
    ret = []
    for m in fileParser.modules.values():
        l = list()
        retained = False
        sum = 0
        for e in m._eventInfo:
            l.append(e._nTemp)
            if e._temp > 0:
                sum += e._nTemp
                retained = True
        if retained:
            if showEvents:
                ret.append((m._label, m._type, float(sum)/len(l), l))
            else:
                ret.append((m._label, m._type, float(sum)/len(l)))
    return ret


def printReport(values, showEvents, summary, eventSummary, maxColumn):
    values.sort(key=lambda x: x[2])
    label = "module label"
    classType = "module class type"
    if maxColumn == 0:
        columnWidths = [len(label),len(classType),len(summary)]
        for v in values:
            for c in (0,1,2):
                if c == 2:
                    width = len(f"{v[c]}:.2f")
                else:
                    width = len(v[c])
                if width > columnWidths[c]:
                    columnWidths[c] = width
    else:
        columnWidths = [maxColumn, maxColumn, maxColumn]
        label = label[:maxColumn]
        classType = classType[:maxColumn]
    print(f"{label:{columnWidths[0]}} {classType:{columnWidths[1]}} {summary:{columnWidths[2]}}")
    if showEvents:
        print(f" [{eventSummary}]")
        
    for v in values:
        label = v[0]
        classType = v[1]
        if maxColumn:
            label = label[:maxColumn]
            classType = classType[:maxColumn]
        print(f"{label:{columnWidths[0]}} {classType:{columnWidths[1]}} {v[2]:{columnWidths[2]}.2f}")
        if showEvents:
            print(f" {v[3]}")

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Parses files generated from ModuleEventAlloc service')
    parser.add_argument('filename',
                        type=argparse.FileType('r'), # open file
                        help='file to process')
    parser.add_argument('--grew', help='report which modules retained more memory over the job', action='store_true')
    parser.add_argument('--retained', help='report which modules retained memory between events (might be deleted at begin of next event)', action='store_true')
    parser.add_argument('--product', help="report how much memory each module put into the Event as data products", action='store_true')
    parser.add_argument('--tempSize', help="report how much temporary allocated memory each module used when processing the Event", action='store_true')
    parser.add_argument('--nTemp', help="report number of temporary allocations each module used when processing the Event", action='store_true')
    
    parser.add_argument('--eventData', help='for each report, show the per event data associated to the report', action='store_true')
    parser.add_argument('--maxColumn', type=int, help='maximum column width for report, 0 for no constraint', default=0)
    args = parser.parse_args()
    
    inputfile = args.filename

    fileParser = FileParser()
    fileParser.parse(inputfile)

    if args.grew:
        printReport(reportModulesWithMemoryGrowth(fileParser, args.eventData), args.eventData, "total memory growth", "growth each event", args.maxColumn)
    if args.retained:
        printReport(reportModuleRetainingMemory(fileParser, args.eventData), args.eventData, "average retained", "retained each event", args.maxColumn)
    if args.product:
        printReport(reportModuleDataProductMemory(fileParser, args.eventData), args.eventData, "average data products size", "data products size each event", args.maxColumn)
    if args.tempSize:
        printReport(reportModuleTemporary(fileParser, args.eventData), args.eventData, "average temporary allocation size", "temporary allocation size each event", args.maxColumn)
    if args.nTemp:
        printReport(reportModuleNTemporary(fileParser, args.eventData), args.eventData, "average # of temporary allocation", "# of temporary allocations each event", args.maxColumn)
    #print(fileParser.modules)
