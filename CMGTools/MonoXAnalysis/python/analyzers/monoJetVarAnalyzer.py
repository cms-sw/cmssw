from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.HeppyCore.utils.deltar import deltaR, deltaPhi

import operator 
import itertools
import copy
from math import *

from CMGTools.RootTools.statistics.Counter import Counter, Counters
from CMGTools.RootTools.fwlite.AutoHandle import AutoHandle

import ROOT

import os

class monoJetVarAnalyzer( Analyzer ):
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(monoJetVarAnalyzer,self).__init__(cfg_ana,cfg_comp,looperName) 

    def declareHandles(self):
        super(monoJetVarAnalyzer, self).declareHandles()

    def beginLoop(self, setup):
        super(monoJetVarAnalyzer,self).beginLoop(setup)
        self.counters.addCounter('pairs')
        count = self.counters.counter('pairs')
        count.register('all events')


    # Calculate some topological variables
    def getApcJetMetMin(self, event):

        if len(event.cleanJets) == 0:
            event.apcjetmetmin = 0
            return
        
        px  = ROOT.std.vector('double')()
        py  = ROOT.std.vector('double')()
        et  = ROOT.std.vector('double')()
        metx = event.metNoMu.px()
        mety = event.metNoMu.py()        

	for jet in event.cleanJets:
            px.push_back(jet.px())
            py.push_back(jet.py())
            et.push_back(jet.et())
            pass

        apcCalc   = ROOT.heppy.Apc()
        event.apcjetmetmin = apcCalc.getApcJetMetMin( et, px, py, metx, mety )

        return

    def process(self, event):
        self.readCollections( event.input )

        event.apc = -999
        self.getApcJetMetMin(event)

        return True
