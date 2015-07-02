from math import *
from CMGTools.HToZZ4L.analyzers.FourLeptonAnalyzer import *

        
class FourLeptonAnalyzerSS( FourLeptonAnalyzer ):
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(FourLeptonAnalyzerSS,self).__init__(cfg_ana,cfg_comp,looperName)
        self.tag = cfg_ana.tag

    def declareHandles(self):
        super(FourLeptonAnalyzerSS, self).declareHandles()

    def beginLoop(self, setup):
        super(FourLeptonAnalyzerSS,self).beginLoop(setup)
        self.counters.addCounter('FourLepton')
        count = self.counters.counter('FourLepton')
        count.register('all events')


    #For the good lepton preselection redefine the thingy so that leptons are loose    
    def leptonID(self,lepton):
        return self.leptonID_loose(lepton)

    def zSorting(self,Z1,Z2):
        return True


    #Redefine the QUADS so Z2 is SF/SS!!!
    def findOSSFQuads(self, leptons,photons):
        '''Make combinatorics and make permulations of four leptons
           Cut the permutations by asking Z1 nearest to Z and also 
           that plus is the first
           Include FSR if in cfg file
        '''
        out = []
        for l1, l2,l3,l4 in itertools.permutations(leptons, 4):
            if (l1.pdgId()+l2.pdgId())!=0: 
                continue;
            if (l3.pdgId()!=l4.pdgId()):
                continue;
            if (l1.pdgId()<l2.pdgId())!=0: 
                continue;
            quadObject =DiObjectPair(l1, l2,l3,l4)
            self.attachFSR(quadObject,photons)
            if not self.zSorting(quadObject.leg1,quadObject.leg2):
                continue;
            out.append(quadObject)

        return out




    def fourLeptonIsolation(self,fourLepton):
        ##Fancy! Here apply only tight ID on Z1 and no ID in Z2
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
        return True        
