from math import *
from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.HeppyCore.utils.deltar import deltaR, deltaPhi
from PhysicsTools.HeppyCore.framework.event import *

from CMGTools.HToZZ4L.tools.DiObject import DiObject

import os
import itertools
import collections
import ROOT

    
class TwoLeptonAnalyzer( Analyzer ):
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(TwoLeptonAnalyzer,self).__init__(cfg_ana,cfg_comp,looperName)


    def beginLoop(self, setup):
        super(TwoLeptonAnalyzer,self).beginLoop(setup)
        self.counters.addCounter('TwoLepton')
        count = self.counters.counter('TwoLepton')
        count.register('all events')
        count.register('all pairs')
        count.register('pass iso')
        count.register('best Z')

    def process(self, event):
        self.readCollections( event.input )

        self.counters.counter('TwoLepton').inc('all events')
        # count tight leptons
        tight_leptons = [ lep for lep in event.selectedLeptons if self.leptonID_tight(lep) ]

        # make dilepton pairs, possibly attach FSR photons (the latter not yet implemented)
        event.allPairs = self.findOSSFPairs(tight_leptons, event.fsrPhotons)

        # count them, for the record
        for p in event.allPairs:
            self.counters.counter('TwoLepton').inc('all pairs')

        # make pairs of isolated leptons
        event.isolatedPairs = filter(self.twoLeptonIsolation, event.allPairs)
        for pair in event.isolatedPairs:
            self.counters.counter('TwoLepton').inc('pass iso')

        # get the best Z (mass closest to PDG value)
        # still a list, because if there's no isolated leptons it may be empty
        sortedIsoPairs = event.isolatedPairs[:] # make a copy
        sortedIsoPairs.sort(key = lambda dilep : abs(dilep.mass() - 91.1876))
        event.bestIsoZ = sortedIsoPairs[:1] # pick at most 1
        if len(event.bestIsoZ):
            self.counters.counter('TwoLepton').inc('best Z')

    def leptonID_tight(self,lepton):
        return lepton.tightId()

    def muonIsolation(self,lepton):
        return lepton.absIsoWithFSR(R=0.4,puCorr="deltaBeta")/lepton.pt()<0.4

    def electronIsolation(self,lepton):
        return lepton.absIsoWithFSR(R=0.4,puCorr="rhoArea")/lepton.pt()<0.5


    def twoLeptonIsolation(self,twoLepton):
        ##First ! attach the FSR photons of this candidate to the leptons!
        
        leptons = twoLepton.daughterLeptons()
        photons = twoLepton.daughterPhotons()

        for l in leptons:
            l.fsrPhotons=[]
            for g in photons:
                if deltaR(g.eta(),g.phi(),l.eta(),l.phi())<0.4:
                    l.fsrPhotons.append(g)
            if abs(l.pdgId())==11:
                if not self.electronIsolation(l):
                    return False
            if abs(l.pdgId())==13:
                if not self.muonIsolation(l):
                    return False
        return True        

    def findOSSFPairs(self, leptons, photons):
        '''Make combinatorics and make permulations of two leptons
           Include FSR if in cfg file
        '''
        out = []
        for l1, l2 in itertools.permutations(leptons, 2):
            if (l1.pdgId()+l2.pdgId())!=0: 
                continue;
            if (l1.pdgId()<l2.pdgId())!=0: 
                continue;

            twoObject = DiObject(l1, l2)
            # ---- FIXME should do FSR recovery here 
            self.attachFSR(twoObject,photons)
            out.append(twoObject)

        return out

    def attachFSR(self,dilep,photons):
        # Not implemented yet
        return True
