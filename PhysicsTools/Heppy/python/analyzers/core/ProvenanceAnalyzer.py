import itertools

from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from PhysicsTools.HeppyCore.framework.event import Event
from PhysicsTools.HeppyCore.statistics.counter import Counter, Counters

from DataFormats.FWLite import Events, Handle,Lumis

class ProvenanceAnalyzer( Analyzer ):
    #---------------------------------------------
    # TO FINDS THE INITIAL EVENTS BEFORE THE SKIM
    #---------------------------------------------
    
    def __init__(self, cfg_ana, cfg_comp, looperName):
        super(ProvenanceAnalyzer, self).__init__(cfg_ana, cfg_comp, looperName)
        self.lastId = (0,0)
        self.provenance = []
 
    def declareHandles(self):
        super(ProvenanceAnalyzer, self).declareHandles()
    
    def cmsswVNums(self, release):
        vpieces = release.split("_")[1:]
        vnums = [ int(v) for v in vpieces[:3] ]
        if len(vpieces) > 3:
            if "patch" in vpieces[3]:
                vnums.append(int(vpieces[3].replace("patch","")))
            elif "pre" in vpieces[3]:
                vnums.append(-100+int(vpieces[3].replace("pre","")))
        else:
            vnums.append(0)
        return tuple(vnums)
    def miniAODVersion(self, vnums):
        if vnums[:2] == (7,4):
            if vnums >= (7,4,14,0): return (2015,2.1)
            if vnums >= (7,4,12,0): return (2015,2.0)
            if vnums >= (7,4, 8,1): return (2015,1.1)
            return (2015,1.0)
        elif vnums[:2] == (7,2):
            return (2014,2.1)   
        elif vnums[:2] == (7,0):
            if vnums >= (7,0,9,1): return (2014,2.0)
            return (2014,1.0)
        else:
            return (-999,-999)

    def process(self, event):
        eid = (  event.input.eventAuxiliary().id().run(),
                 event.input.eventAuxiliary().id().luminosityBlock() )
        if eid != self.lastId:
            #import pdb; pdb.set_trace()
            history = event.input.object().processHistory()
            for i in reversed(range(history.size())):
                conf = history.at(i)
                release = conf.releaseVersion().replace('"',"")
                vnums = self.cmsswVNums(release)
                if conf.processName() in  ("PAT", "RECO","REC2"):
                    print "processing step %d: process %s, release %r %s, miniAOD %s" % (i, conf.processName(), release, vnums, self.miniAODVersion(vnums))
                    self.provenance = [
                        ('release', release),
                        ('releaseVersion', vnums),
                        ('miniAODVersion', self.miniAODVersion(vnums)),
                    ]
                    break
            self.lastId = eid
        for key,value in self.provenance:
            setattr(event, key, value)
        return True

import PhysicsTools.HeppyCore.framework.config as cfg
setattr(ProvenanceAnalyzer,"defaultConfig", cfg.Analyzer(
    class_object = ProvenanceAnalyzer
))

