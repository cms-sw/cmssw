from math import *
from CMGTools.HToZZ4L.analyzers.FourLeptonAnalyzer import *

        
class FourLeptonAnalyzer3P1F( FourLeptonAnalyzer ):
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(FourLeptonAnalyzer3P1F,self).__init__(cfg_ana,cfg_comp,looperName)
        self.tag = cfg_ana.tag
    def declareHandles(self):
        super(FourLeptonAnalyzer3P1F, self).declareHandles()

    def beginLoop(self, setup):
        super(FourLeptonAnalyzer3P1F,self).beginLoop(setup)
        self.counters.addCounter('FourLepton')
        count = self.counters.counter('FourLepton')
        count.register('all events')


    #For the good lepton preselection redefine the thingy so that leptons are loose    
    def leptonID(self,lepton):
        return self.leptonID_loose(lepton)


    def zSorting(self,Z1,Z2):
        return True

    def fourLeptonIsolation(self,fourLepton):
        ##Fancy! Here require that Z1 leptons pass tight ID and isolationand the two other leptons fail ID or isolation
        leptons = fourLepton.daughterLeptons()
        photons = fourLepton.daughterPhotons()
        for l in [fourLepton.leg1.leg1,fourLepton.leg1.leg2]:
            if not self.leptonID_tight(l):
                return False
            l.fsrPhotons=[]
            for g in photons:
                if deltaR(g.eta(),g.phi(),l.eta(),l.phi())<0.4:
                    l.fsrPhotons.append(g)
            if abs(l.pdgId())==11:
                if not (self.electronIsolation(l)):
                    return False
            if abs(l.pdgId())==13:
                if not self.muonIsolation(l):
                    return False

        failed=0        
        for l in [fourLepton.leg2.leg1,fourLepton.leg2.leg2]:
            if  not self.leptonID_tight(l):
                failed=failed+1
                continue 
            l.fsrPhotons=[]
            for g in photons:
                if deltaR(g.eta(),g.phi(),l.eta(),l.phi())<0.4:
                    l.fsrPhotons.append(g)
            if abs(l.pdgId())==11:
                if  not self.electronIsolation(l):
                    failed=failed+1
                    continue
            if abs(l.pdgId())==13:
                if  not self.muonIsolation(l):
                    failed=failed+1
                    continue
        if failed==1:
            return True        
        else:
            return False
