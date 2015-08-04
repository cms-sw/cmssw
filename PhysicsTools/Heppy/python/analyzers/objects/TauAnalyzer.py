from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from PhysicsTools.Heppy.physicsobjects.Tau import Tau

from PhysicsTools.HeppyCore.utils.deltar import deltaR, matchObjectCollection3

import PhysicsTools.HeppyCore.framework.config as cfg

 
class TauAnalyzer( Analyzer ):

    
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(TauAnalyzer,self).__init__(cfg_ana,cfg_comp,looperName)

    #----------------------------------------
    # DECLARATION OF HANDLES OF LEPTONS STUFF   
    #----------------------------------------
    def declareHandles(self):
        super(TauAnalyzer, self).declareHandles()
        self.handles['taus'] = AutoHandle( ('slimmedTaus',''),'std::vector<pat::Tau>')

    def beginLoop(self, setup):
        super(TauAnalyzer,self).beginLoop(setup)
        self.counters.addCounter('events')
        count = self.counters.counter('events')
        count.register('all events')
        count.register('has >=1 tau at preselection')
        count.register('has >=1 selected taus')
        count.register('has >=1 other taus')

    #------------------
    # MAKE LEPTON LISTS
    #------------------
    def makeTaus(self, event):
        event.inclusiveTaus = []
        event.selectedTaus = []
        event.otherTaus = []

        #get all
        alltaus = map( Tau, self.handles['taus'].product() )

        #make inclusive taus
        for tau in alltaus:
            tau.associatedVertex = event.goodVertices[0] if len(event.goodVertices)>0 else event.vertices[0]
            tau.lepVeto = False
            tau.idDecayMode = tau.tauID("decayModeFinding")
            tau.idDecayModeNewDMs = tau.tauID("decayModeFindingNewDMs")

            if hasattr(self.cfg_ana, 'inclusive_decayModeID') and self.cfg_ana.inclusive_decayModeID and not tau.tauID(self.cfg_ana.inclusive_decayModeID):
                continue

            tau.inclusive_lepVeto = False
            if self.cfg_ana.inclusive_vetoLeptons:
                for lep in event.selectedLeptons:
                    if deltaR(lep.eta(), lep.phi(), tau.eta(), tau.phi()) < self.cfg_ana.inclusive_leptonVetoDR:
                        tau.inclusive_lepVeto = True
                if tau.inclusive_lepVeto: continue
            if self.cfg_ana.inclusive_vetoLeptonsPOG:
                if not tau.tauID(self.cfg_ana.inclusive_tauAntiMuonID):
                        tau.inclusive_lepVeto = True
                if not tau.tauID(self.cfg_ana.inclusive_tauAntiElectronID):
                        tau.inclusive_lepVeto = True
                if tau.inclusive_lepVeto: continue

            if tau.pt() < self.cfg_ana.inclusive_ptMin: continue
            if abs(tau.eta()) > self.cfg_ana.inclusive_etaMax: continue
            if abs(tau.dxy()) > self.cfg_ana.inclusive_dxyMax or abs(tau.dz()) > self.cfg_ana.inclusive_dzMax: continue

            def id3(tau,X):
                """Create an integer equal to 1-2-3 for (loose,medium,tight)"""
                return tau.tauID(X%"Loose") + tau.tauID(X%"Medium") + tau.tauID(X%"Tight")
            def id5(tau,X):
                """Create an integer equal to 1-2-3-4-5 for (very loose, 
                    loose, medium, tight, very tight)"""
                return id3(tau, X) + tau.tauID(X%"VLoose") + tau.tauID(X%"VTight")
            def id6(tau,X):
                """Create an integer equal to 1-2-3-4-5-6 for (very loose, 
                    loose, medium, tight, very tight, very very tight)"""
                return id5(tau, X) + tau.tauID(X%"VVTight")

            tau.idMVA = id6(tau, "by%sIsolationMVA3oldDMwLT")
            tau.idMVANewDM = id6(tau, "by%sIsolationMVA3newDMwLT")
            tau.idCI3hit = id3(tau, "by%sCombinedIsolationDeltaBetaCorr3Hits")
            tau.idAntiMu = tau.tauID("againstMuonLoose") + tau.tauID("againstMuonTight")
            tau.idAntiE = id5(tau, "againstElectron%sMVA5")
            #print "Tau pt %5.1f: idMVA2 %d, idCI3hit %d, %s, %s" % (tau.pt(), tau.idMVA2, tau.idCI3hit, tau.tauID(self.cfg_ana.tauID), tau.tauID(self.cfg_ana.tauLooseID))
            
            if tau.tauID(self.cfg_ana.inclusive_tauID):
                event.inclusiveTaus.append(tau)
            
        for tau in event.inclusiveTaus:

            tau.loose_lepVeto = False
            if self.cfg_ana.loose_vetoLeptons:
                for lep in event.selectedLeptons:
                    if deltaR(lep.eta(), lep.phi(), tau.eta(), tau.phi()) < self.cfg_ana.loose_leptonVetoDR:
                        tau.loose_lepVeto = True
            if self.cfg_ana.loose_vetoLeptonsPOG:
                if not tau.tauID(self.cfg_ana.loose_tauAntiMuonID):
                        tau.loose_lepVeto = True
                if not tau.tauID(self.cfg_ana.loose_tauAntiElectronID):
                        tau.loose_lepVeto = True

            if tau.tauID(self.cfg_ana.loose_decayModeID) and \
                      tau.pt() > self.cfg_ana.loose_ptMin and abs(tau.eta()) < self.cfg_ana.loose_etaMax and \
                      abs(tau.dxy()) < self.cfg_ana.loose_dxyMax and abs(tau.dz()) < self.cfg_ana.loose_dzMax and \
                      tau.tauID(self.cfg_ana.loose_tauID) and not tau.loose_lepVeto:
                event.selectedTaus.append(tau)
            else:
                event.otherTaus.append(tau)

        event.inclusiveTaus.sort(key = lambda l : l.pt(), reverse = True)        
        event.selectedTaus.sort(key = lambda l : l.pt(), reverse = True)
        event.otherTaus.sort(key = lambda l : l.pt(), reverse = True)
        self.counters.counter('events').inc('all events')
        if len(event.inclusiveTaus): self.counters.counter('events').inc('has >=1 tau at preselection')
        if len(event.selectedTaus): self.counters.counter('events').inc('has >=1 selected taus')
        if len(event.otherTaus): self.counters.counter('events').inc('has >=1 other taus')

    def matchTaus(self, event):
        match = matchObjectCollection3(event.inclusiveTaus, event.gentaus, deltaRMax = 0.5)
        for lep in event.inclusiveTaus:
            gen = match[lep]
            lep.mcMatchId = 1 if gen else 0
            lep.genp = gen

    def process(self, event):
        self.readCollections( event.input )

        self.makeTaus(event)

        if not self.cfg_comp.isMC:
            return True

        if hasattr(event, 'gentaus'):
            self.matchTaus(event)
        
        return True

# Find the definitions of the tau ID strings here:
# http://cmslxr.fnal.gov/lxr/source/PhysicsTools/PatAlgos/python/producersLayer1/tauProducer_cfi.py

setattr(TauAnalyzer,"defaultConfig",cfg.Analyzer(
    class_object = TauAnalyzer,
    # inclusive very loose hadronic tau selection
    inclusive_ptMin = 18,
    inclusive_etaMax = 9999,
    inclusive_dxyMax = 1000.,
    inclusive_dzMax = 0.4,
    inclusive_vetoLeptons = False,
    inclusive_leptonVetoDR = 0.4,
    inclusive_decayModeID = "decayModeFindingNewDMs", # ignored if not set or ""
    inclusive_tauID = "decayModeFindingNewDMs",
    inclusive_vetoLeptonsPOG = False, # If True, the following two IDs are required
    inclusive_tauAntiMuonID = "",
    inclusive_tauAntiElectronID = "",
    # loose hadronic tau selection
    loose_ptMin = 18,
    loose_etaMax = 9999,
    loose_dxyMax = 1000.,
    loose_dzMax = 0.2,
    loose_vetoLeptons = True,
    loose_leptonVetoDR = 0.4,
    loose_decayModeID = "decayModeFindingNewDMs", # ignored if not set or ""
    loose_tauID = "byLooseCombinedIsolationDeltaBetaCorr3Hits",
    loose_vetoLeptonsPOG = False, # If True, the following two IDs are required
    loose_tauAntiMuonID = "againstMuonLoose3",
    loose_tauAntiElectronID = "againstElectronLooseMVA5",
    loose_tauLooseID = "decayModeFindingNewDMs"
  )
)
