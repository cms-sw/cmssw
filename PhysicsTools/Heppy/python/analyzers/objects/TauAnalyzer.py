
import operator 
import itertools
import copy
import types

from ROOT import TLorentzVector

from CMGTools.RootTools.fwlite.Analyzer import Analyzer
from CMGTools.RootTools.fwlite.Event import Event
from CMGTools.RootTools.statistics.Counter import Counter, Counters
from CMGTools.RootTools.fwlite.AutoHandle import AutoHandle
from CMGTools.RootTools.physicsobjects.Lepton import Lepton
from CMGTools.RootTools.physicsobjects.Tau import Tau

from CMGTools.RootTools.utils.DeltaR import deltaR, deltaPhi, bestMatch
 
class ttHTauAnalyzer( Analyzer ):

    
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(ttHTauAnalyzer,self).__init__(cfg_ana,cfg_comp,looperName)

    #----------------------------------------
    # DECLARATION OF HANDLES OF LEPTONS STUFF   
    #----------------------------------------
    def declareHandles(self):
        super(ttHTauAnalyzer, self).declareHandles()
        self.handles['taus'] = AutoHandle( ('slimmedTaus',''),'std::vector<pat::Tau>')


    def beginLoop(self):
        super(ttHTauAnalyzer,self).beginLoop()
        self.counters.addCounter('events')
        count = self.counters.counter('events')
        count.register('all events')
        count.register('has >=1 tau at preselection')
        count.register('has >=1 selected taus')
        count.register('has >=1 loose taus')
        count.register('has >=1 inclusive taus')

    #------------------
    # MAKE LEPTON LISTS
    #------------------
    def makeTaus(self, event):
        event.selectedTaus = []
        event.looseTaus = []
        event.inclusiveTaus = []

        #get all
        alltaus = map( Tau, self.handles['taus'].product() )

        foundTau = False
        for tau in alltaus:
            tau.associatedVertex = event.goodVertices[0]
            tau.lepVeto = False
            if self.cfg_ana.vetoLeptons:
                for lep in event.selectedLeptons:
                    if deltaR(lep.eta(), lep.phi(), tau.eta(), tau.phi()) < self.cfg_ana.leptonVetoDR:
                        tau.lepVeto = True
                if tau.lepVeto: continue
            if self.cfg_ana.vetoLeptonsPOG:
                if not tau.tauID("againstMuonTight"):
                        tau.lepVeto = True
                if not tau.tauID("againstElectronLoose"):
                        tau.lepVeto = True
                if tau.lepVeto: continue
            if tau.pt() < self.cfg_ana.ptMin: continue
            if abs(tau.eta()) > self.cfg_ana.etaMax: continue
###            tau.dxy and tau.dz are zero
###            if abs(tau.dxy()) > self.cfg_ana.dxyMax or abs(tau.dz()) > self.cfg_ana.dzMax: continue
            foundTau = True
            def id3(tau,X):
                """Create an integer equal to 1-2-3 for (loose,medium,tight)"""
                return tau.tauID(X%"Loose") + tau.tauID(X%"Medium") + tau.tauID(X%"Tight")
            #tau.idMVA2   = id3(tau, "by%sIsolationMVA2")
            tau.idCI3hit = id3(tau, "by%sCombinedIsolationDeltaBetaCorr3Hits")
            #print "Tau pt %5.1f: idMVA2 %d, idCI3hit %d, %s, %s" % (tau.pt(), tau.idMVA2, tau.idCI3hit, tau.tauID(self.cfg_ana.tauID), tau.tauID(self.cfg_ana.tauLooseID))
            if tau.tauID(self.cfg_ana.tauID):
                event.selectedTaus.append(tau)
                event.inclusiveTaus.append(tau)
            elif tau.tauID(self.cfg_ana.tauLooseID):
                event.looseTaus.append(tau)
                event.inclusiveTaus.append(tau)

        event.selectedTaus.sort(key = lambda l : l.pt(), reverse = True)
        event.looseTaus.sort(key = lambda l : l.pt(), reverse = True)
        self.counters.counter('events').inc('all events')
        if foundTau: self.counters.counter('events').inc('has >=1 tau at preselection')
        if len(event.selectedTaus): self.counters.counter('events').inc('has >=1 selected taus')
        if len(event.looseTaus): self.counters.counter('events').inc('has >=1 loose taus')
        if len(event.inclusiveTaus): self.counters.counter('events').inc('has >=1 inclusive taus')
    def process(self, iEvent, event):
        self.readCollections( iEvent )
        self.makeTaus(event)
        return True
