import operator 
import itertools
import copy
from math import *

from ROOT import std 
from ROOT import TLorentzVector, TVectorD

from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.HeppyCore.framework.event import Event
from PhysicsTools.HeppyCore.statistics.counter import Counter, Counters
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle

from PhysicsTools.HeppyCore.utils.deltar import deltaR


import ROOT

import os


class ttHJZBTopologicalVars( Analyzer ):
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(ttHJZBTopologicalVars,self).__init__(cfg_ana,cfg_comp,looperName) 

    def declareHandles(self):
        super(ttHJZBTopologicalVars, self).declareHandles()
       #genJets                                                                                                                                                                     
        self.handles['genJets'] = AutoHandle( 'slimmedGenJets','std::vector<reco::GenJet>')

    def beginLoop(self, setup):
        super(ttHJZBTopologicalVars,self).beginLoop(setup)
        self.counters.addCounter('pairs')
        count = self.counters.counter('pairs')
        count.register('all events')

    def makeMETRecoil(self, event):
        
        if len(event.selectedLeptons) > 1:
            event.METRecoil = event.met.p4() + event.selectedLeptons[0].p4() + event.selectedLeptons[1].p4() 


    def makeHadronicRecoil(self, event):
        objectsjet40 = [ j for j in event.cleanJets if j.pt() > 40 and abs(j.eta())<3.0 ]
        if len(objectsjet40)>0:
            for jet in objectsjet40:
                event.HadronicRecoil = event.HadronicRecoil + jet.p4()
  	  
    def makeJZB(self, event):
        
        if len(event.selectedLeptons) > 1:
           event.jzb = event.METRecoil.pt() - (event.selectedLeptons[0].p4() + event.selectedLeptons[1].p4()).pt()        
        
    def makeZVars(self, event):

        if len(event.selectedLeptons) > 1:
            event.l1l2_m = (event.selectedLeptons[0].p4() + event.selectedLeptons[1].p4()).M() 
            event.l1l2_pt = (event.selectedLeptons[0].p4() + event.selectedLeptons[1].p4()).pt() 
            event.l1l2_eta = (event.selectedLeptons[0].p4() + event.selectedLeptons[1].p4()).eta() 
            event.l1l2_phi = (event.selectedLeptons[0].p4() + event.selectedLeptons[1].p4()).phi() 
            event.l1l2_DR = deltaR(event.selectedLeptons[0].eta(), event.selectedLeptons[0].phi(), event.selectedLeptons[1].eta(), event.selectedLeptons[1].phi())  

    def makeZGenVars(self, event):

        if len(event.genleps) > 1:
            event.genl1l2_m = (event.genleps[0].p4() + event.genleps[1].p4()).M() 
            event.genl1l2_pt = (event.genleps[0].p4() + event.genleps[1].p4()).pt() 
            event.genl1l2_eta = (event.genleps[0].p4() + event.genleps[1].p4()).eta() 
            event.genl1l2_phi = (event.genleps[0].p4() + event.genleps[1].p4()).phi() 
            event.genl1l2_DR = deltaR(event.genleps[0].eta(), event.genleps[0].phi(), event.genleps[1].eta(), event.genleps[1].phi())  

    def process(self, event):
        self.readCollections( event.input )
        
        event.l1l2_m = 0
        event.l1l2_pt = 0
        event.l1l2_eta = 0
        event.l1l2_phi = 0
        event.l1l2_DR = 0
        
        event.genl1l2_m = 0
        event.genl1l2_pt = 0
        event.genl1l2_eta = 0
        event.genl1l2_phi = 0
        event.genl1l2_DR = 0
   
        event.jzb = 0
        event.HadronicRecoil = ROOT.reco.Particle.LorentzVector( 0, 0, 0, 0 )
        event.METRecoil = ROOT.reco.Particle.LorentzVector( 0, 0, 0, 0 )

        self.makeZVars(event)
        self.makeZGenVars(event)
        self.makeMETRecoil(event)
        self.makeHadronicRecoil(event)
        self.makeJZB(event)

        return True
