import operator 
import itertools
import copy
from math import *

#from ROOT import TLorentzVector, TVectorD

from PhysicsTools.HeppyCore.utils.deltar import deltaR
from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.HeppyCore.framework.event import Event
from PhysicsTools.HeppyCore.statistics.counter import Counter, Counters
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle

# from PhysicsTools.Heppy.physicsobjects.PhysicsObjects import Lepton
# from PhysicsTools.Heppy.physicsobjects.PhysicsObjects import Photon
# from PhysicsTools.Heppy.physicsobjects.PhysicsObjects import Electron
# from PhysicsTools.Heppy.physicsobjects.PhysicsObjects import Muon
# from PhysicsTools.Heppy.physicsobjects.PhysicsObjects import Tau
from PhysicsTools.Heppy.physicsobjects.PhysicsObjects import Jet

import os

# Function to calculate the transverse mass
def mtw(x1,x2):
    return sqrt(2*x1.pt()*x2.pt()*(1-cos(x1.phi()-x2.phi())))

class ttHAlphaTControlAnalyzer( Analyzer ):
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(ttHAlphaTControlAnalyzer,self).__init__(cfg_ana,cfg_comp,looperName) 

        self.maxLeps = cfg_ana.maxLeps if hasattr(cfg_ana,'maxLeps') else 999
        self.maxPhotons = cfg_ana.maxPhotons if hasattr(cfg_ana,'maxPhotons') else 999

    def declareHandles(self):
        super(ttHAlphaTControlAnalyzer, self).declareHandles()
       #genJets                                                                                                                                                                     
        self.handles['genJets'] = AutoHandle( 'slimmedGenJets','std::vector<reco::GenJet>')

    def beginLoop(self,setup):
        super(ttHAlphaTControlAnalyzer,self).beginLoop(setup)
        self.counters.addCounter('pairs')
        count = self.counters.counter('pairs')
        count.register('all events')


    # Calculate MT_W (stolen from the MT2 code)
    # Modularize this later?
    # Does it just for the leading lepton 
    def makeMT(self, event):
    # print '==> INSIDE THE PRINT MT'
    # print 'MET=',event.met.pt() 

        if len(event.selectedLeptons)>0:
            event.mtw = mtw(event.selectedLeptons[0], event.met)

        if len(event.selectedTaus)>0:
            event.mtwTau = mtw(event.selectedTaus[0], event.met)
                
        if len(event.selectedIsoTrack)>0:
            event.mtwIsoTrack = mtw(event.selectedIsoTrack[0], event.met)

        return

    # Calculate the invariant mass from two lead leptons
    def makeMll(self, event):
        
        if len(event.selectedLeptons)>=2:
            event.mll = (event.selectedLeptons[0].p4()+event.selectedLeptons[1].p4()).M()

        return

    # Calculate the DeltaR between the lepton and the closest jet
    def makeDeltaRLepJet(self, event):

        event.minDeltaRLepJet = []

        for i,lepton in enumerate(event.selectedLeptons):

            if i == self.maxLeps: break

            minDeltaR = 999

            for jet in event.cleanJets:
                minDeltaR=min(deltaR(lepton.eta(),lepton.phi(),jet.eta(),jet.phi()), minDeltaR)

            # Fill event with the min deltaR for each lepton
            event.minDeltaRLepJet.append(minDeltaR)

        return

    # Calculate the DeltaR between the photon and the closest jet
    def makeDeltaRPhoJet(self, event):

        event.minDeltaRPhoJet = []

        for i,photon in enumerate(event.selectedPhotons):

            if i == self.maxPhotons: break

            minDeltaR = 999

            for jet in event.cleanJets:
                minDeltaR=min(deltaR(photon.eta(),photon.phi(),jet.eta(),jet.phi()), minDeltaR)

            # Fill event with the min deltaR for each photon
            event.minDeltaRPhoJet.append(minDeltaR)

        return

    def process(self, event):
        self.readCollections( event.input )

        #W variables
        event.mtw = -999
        event.mtwTau = -999
        event.mtwIsoTrack = -999
        self.makeMT(event)

        #Z variables
        event.mll = -999
        self.makeMll(event)

        #Delta R variables
        self.makeDeltaRLepJet(event)
        self.makeDeltaRPhoJet(event)

        return True
