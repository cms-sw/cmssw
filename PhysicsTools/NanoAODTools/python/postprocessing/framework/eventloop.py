from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Event
from PhysicsTools.NanoAODTools.postprocessing.framework.treeReaderArrayTools import clearExtraBranches
import sys
import time
import ROOT


class Module(object):
    def __init__(self):
        self.writeHistFile = False

    def beginJob(self, histFile=None, histDirName=None):
        if histFile != None and histDirName != None:
            self.writeHistFile = True
            prevdir = ROOT.gDirectory
            self.histFile = histFile
            self.histFile.cd()
            self.dir = self.histFile.mkdir(histDirName)
            prevdir.cd()
            self.objs = []

    def endJob(self):
        if hasattr(self, 'objs') and self.objs != None:
            prevdir = ROOT.gDirectory
            self.dir.cd()
            for obj in self.objs:
                obj.Write()
            prevdir.cd()
            if hasattr(self, 'histFile') and self.histFile != None:
                self.histFile.Close()

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        pass

    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        pass

    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""
        pass

    def addObject(self, obj):
        setattr(self, obj.GetName(), obj)
        self.objs.append(getattr(self, obj.GetName()))

    def addObjectList(self, names, obj):
        objlist = []
        for iname, name in enumerate(names):
            setattr(self, obj.GetName() + '_' + name,
                    obj.Clone(obj.GetName() + '_' + name))
            objlist.append(getattr(self, obj.GetName() + '_' + name))
            self.objs.append(getattr(self, obj.GetName() + '_' + name))
        setattr(self, obj.GetName(), objlist)


def eventLoop(
        modules, inputFile, outputFile, inputTree, wrappedOutputTree,
        maxEvents=-1, eventRange=None, progress=(10000, sys.stdout),
        filterOutput=True
):
    for m in modules:
        m.beginFile(inputFile, outputFile, inputTree, wrappedOutputTree)

    t0 = time.time()
    tlast = t0
    doneEvents = 0
    acceptedEvents = 0
    entries = inputTree.entries
    if eventRange:
        entries = len(eventRange)
    if maxEvents > 0:
        entries = min(entries, maxEvents)

    for ie, i in enumerate(range(entries) if eventRange == None else eventRange):
        if maxEvents > 0 and ie >= maxEvents:
            break
        e = Event(inputTree, i)
        clearExtraBranches(inputTree)
        doneEvents += 1
        ret = True
        for m in modules:
            ret = m.analyze(e)
            if not ret:
                break
        if ret:
            acceptedEvents += 1
        if (ret or not filterOutput) and wrappedOutputTree != None:
            wrappedOutputTree.fill()
        if progress:
            if ie > 0 and ie % progress[0] == 0:
                t1 = time.time()
                progress[1].write("Processed %8d/%8d entries, %5.2f%% (elapsed time %7.1fs, curr speed %8.3f kHz, avg speed %8.3f kHz), accepted %8d/%8d events (%5.2f%%)\n" % (
                    ie, entries, ie / float(0.01 * entries),
                    t1 - t0, (progress[0] / 1000.) / (max(t1 - tlast, 1e-9)),
                    ie / 1000. / (max(t1 - t0, 1e-9)),
                    acceptedEvents, doneEvents,
                    acceptedEvents / (0.01 * doneEvents)))
                tlast = t1
    for m in modules:
        m.endFile(inputFile, outputFile, inputTree, wrappedOutputTree)
    return (doneEvents, acceptedEvents, time.time() - t0)
