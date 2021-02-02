from __future__ import print_function
#==============================
#
# First argument is a log file from cmsRun
# containing output from Tracer service with
# the configuration containing
#     dumpPathsAndConsumes = cms.untracked.bool(True)
#
# A new configuration will be created with the same
# topology as the original configuration but with
# the modules replaced with 'trivial' versions.
# This allows studying the cost of the framework infrastructure
# on realistic module topologies.
#==============================

import sys
import six

f = open(sys.argv[1])


def fixName(name):
    return name.replace("_","IoI")

class PathParser(object):
    def __init__(self):
        self._pathToModules = dict()
        self._isEndPath = set()
        self._presentPath = []
        self._presentPathName = None
        self.__preamble = 'modules on '
    def parse(self,line):
        if line[:len(self.__preamble)] == self.__preamble:
            if self._presentPathName:
                self._pathToModules[self._presentPathName] = self._presentPath
            self._presentPathName = line.split(" ")[3][:-2]
            if -1 != line.find('end path'):
                self._isEndPath.add(self._presentPathName)
            self._presentPath = []
        else:
            n = line.strip()
            if self._presentPathName != n:
                self._presentPath.append( fixName(n) )
    def finish(self):
        if self._presentPathName:
            self._pathToModules[self._presentPathName] = self._presentPath

class ConsumesParser(object):
    def __init__(self):
        self._consumesForModule = dict()
        self._isAnalyzer = set()
        self._presentConsumes = []
        self._presentModuleName = None
        self.__preramble = '    '
    def parse(self,line):
        if line[:len(self.__preramble)] != self.__preramble:
            if self._presentModuleName:
                self._consumesForModule[self._presentModuleName] = self._presentConsumes
            start = line.find("'")+1
            length = line[start:].find("'")
            self._presentModuleName = fixName(line[start:length+start])
            self._presentConsumes = []
            if -1 != l.find("Analyzer"):
                self._isAnalyzer.add(self._presentModuleName)
        else:
            self._presentConsumes.append( fixName(line[line.find("'")+1:-2]) )
    def finish(self):
        if self._presentModuleName:
            self._consumesForModule[self._presentModuleName] = self._presentConsumes
    
pathParser = PathParser()
consumesParser = ConsumesParser()

parser = pathParser

foundPaths = False
pathStartsWith = "modules on "

startOfConsumes = "All modules and modules in the current process whose products they consume:"
skipLineAfterConsumes = False
doneWithPaths = False

endOfConsumes = "All modules (listed by class and label) and all their consumed products."
for l in f.readlines():
    if not foundPaths:
        if l[:len(pathStartsWith)] == pathStartsWith:
            foundPaths = True
        else:
            #skip lines till find paths
            continue
    if not doneWithPaths:
        if l[:len(startOfConsumes)] == startOfConsumes:
            skipLineAfterConsumes = True
            doneWithPaths = True
            pathParser.finish()
            parser = consumesParser
            continue
    if skipLineAfterConsumes:
        skipLineAfterConsumes = False
        continue
    if l[:len(endOfConsumes)] == endOfConsumes:
        break
    parser.parse(l)

parser.finish()

print("import FWCore.ParameterSet.Config as cms")
print("process = cms.Process('RECO')")

print("""process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(2000))
process.options = cms.untracked.PSet(
#    numberOfThreads = cms.untracked.uint32(8),
    numberOfThreads = cms.untracked.uint32(1),
    numberOfStreams = cms.untracked.uint32(0),
#    wantSummary = cms.untracked.bool(True)
)

process.add_(cms.Service("Timing", summaryOnly = cms.untracked.bool(True)))

# The following two lines reduce the clutter of repeated printouts
# of the same exception message.
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.MessageLogger.cerr.enableStatistics = False

process.MessageLogger.cerr.FwkReport.reportEvery = 50000
process.MessageLogger.cerr.threshold = 'WARNING'
""")

print("process.source = cms.Source('EmptySource')")

allModules = set()
modulesWithConsumes = set()
#needed to get rid of PathStatus modules at end of paths
pathNamesAsModules = set( (fixName(n) for n in pathParser._pathToModules.iterkeys()) )

for m,c in six.iteritems(consumesParser._consumesForModule):
    if m in pathNamesAsModules:
        continue
    if m in consumesParser._isAnalyzer:
        print("process.%s = cms.EDAnalyzer('MultipleIntsAnalyzer', getFromModules = cms.untracked.VInputTag(*[%s]))"%(m,",".join(["cms.InputTag('%s')"%i for i in (n for n in c if n != 'TriggerResults')])))
    elif not c:
        print("process.%s = cms.EDProducer('IntProducer', ivalue = cms.int32(1))"%m)
    else:
        print("process.%s = cms.EDProducer('AddIntsProducer', labels = cms.VInputTag(*[%s]))"%(m,",".join(["'%s'"%i for i in (n for n in c if n != 'TriggerResults')])))
    allModules.add(m)
    for o  in c:
        allModules.add(o)
    modulesWithConsumes.add(m)

for m in six.itervalues(pathParser._pathToModules):
    for i in m:
        allModules.add(i)

for m in allModules.difference(modulesWithConsumes):
    print("process.%s = cms.EDProducer('IntProducer', ivalue = cms.int32(1))"%(m))


print('t = cms.Task(*[%s])'%(",".join(["process.%s"%i for i in allModules if i not in consumesParser._isAnalyzer])))
for p,m in six.iteritems(pathParser._pathToModules):
    if p in pathParser._isEndPath:
        print("process.%s = cms.EndPath(%s)"%(p,"+".join(["process.%s"%i for i in m])))
    else:
        if m:
            print("process.%s = cms.Path(%s,t)"%(p,"+".join(["process.%s"%i for i in m])))
        else:
            print("process.%s = cms.Path()"%(p))
    

#print "paths = ",pathParser._pathToModules
#print "modulesToConsumes =",consumesParser._consumesForModule


