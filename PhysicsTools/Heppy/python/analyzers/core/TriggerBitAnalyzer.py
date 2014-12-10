import operator 
import itertools
import copy
from math import *
import ROOT
from ROOT.heppy import TriggerBitChecker

from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.HeppyCore.framework.event import Event
from PhysicsTools.HeppyCore.statistics.counter import Counter, Counters
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from PhysicsTools.Heppy.analyzers.core.AutoFillTreeProducer  import *
        
class TriggerBitAnalyzer( Analyzer ):
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(TriggerBitAnalyzer,self).__init__(cfg_ana,cfg_comp,looperName)
        if hasattr(self.cfg_ana,"processName"):
                self.processName = self.cfg_ana.processName
        else :
                self.processName = 'HLT'

        if hasattr(self.cfg_ana,"outprefix"):
                self.outprefix = self.cfg_ana.outprefix
        else :
                self.outprefix = self.processName

    def declareHandles(self):
        super(TriggerBitAnalyzer, self).declareHandles()
        self.handles['TriggerResults'] = AutoHandle( ('TriggerResults','',self.processName), 'edm::TriggerResults' )

    def beginLoop(self, setup):
        super(TriggerBitAnalyzer,self).beginLoop(setup)
        self.triggerBitCheckers = []
        for T, TL in self.cfg_ana.triggerBits.iteritems():
                trigVec = ROOT.vector(ROOT.string)()
                for TP in TL:
                    trigVec.push_back(TP)
                outname="%s_%s"%(self.outprefix,T)
                if not hasattr(setup ,"globalVariables") :
                        setup.globalVariables = []
                setup.globalVariables.append( NTupleVariable(outname, lambda ev : getattr(ev,outname), help="OR of %s"%TL) )
                self.triggerBitCheckers.append( (T, TriggerBitChecker(trigVec)) )

    def process(self, event):
        self.readCollections( event.input )
        triggerResults = self.handles['TriggerResults'].product()
        for T,TC in self.triggerBitCheckers:
            outname="%s_%s"%(self.outprefix,T)
            setattr(event,outname, TC.check(event.input.object(), triggerResults))

        return True

