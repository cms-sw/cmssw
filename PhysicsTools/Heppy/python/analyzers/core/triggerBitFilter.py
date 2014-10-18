import operator 
import itertools
import copy
from math import *

from ROOT import TriggerBitChecker

from PhysicsTools.HeppyCore.framework.analyzer import Analyzer
from PhysicsTools.HeppyCore.framework.Event import Event
from PhysicsTools.HeppyCore.statistics.counter import Counter, Counters
from PhysicsTools.Heppy.analyzers.autohandle import AutoHandle
        
class triggerBitFilter( Analyzer ):
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(triggerBitFilter,self).__init__(cfg_ana,cfg_comp,looperName)
        triggers = cfg_comp.triggers
        self.autoAccept = True if len(triggers) == 0 else False
        vetoTriggers = cfg_comp.vetoTriggers if hasattr(cfg_comp, 'vetoTriggers') else []
        import ROOT
        trigVec = ROOT.vector(ROOT.string)()
        for t in triggers: trigVec.push_back(t)
        self.mainFilter = TriggerBitChecker(trigVec)
        if len(vetoTriggers):
            vetoVec = ROOT.vector(ROOT.string)()
            for t in vetoTriggers: vetoVec.push_back(t)
            self.vetoFilter = TriggerBitChecker(vetoVec)
        else:
            self.vetoFilter = None 
        
    def declareHandles(self):
        super(triggerBitFilter, self).declareHandles()
        self.handles['TriggerResults'] = AutoHandle( ('TriggerResults','','HLT'), 'edm::TriggerResults' )

    def beginLoop(self):
        super(triggerBitFilter,self).beginLoop()

    def process(self, iEvent, event):
        if self.autoAccept: return True
        self.readCollections( iEvent )
        if not self.mainFilter.check(iEvent.object(), self.handles['TriggerResults'].product()):
            return False
        if self.vetoFilter != None and self.vetoFilter.check(iEvent.object(), self.handles['TriggerResults'].product()):
            return False
        return True

