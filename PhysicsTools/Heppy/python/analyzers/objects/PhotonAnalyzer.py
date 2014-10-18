import operator
import itertools
import copy
import types

from ROOT import TLorentzVector

from CMGTools.RootTools.fwlite.Analyzer import Analyzer
from CMGTools.RootTools.fwlite.Event import Event
from CMGTools.RootTools.statistics.Counter import Counter, Counters
from CMGTools.RootTools.fwlite.AutoHandle import AutoHandle
from CMGTools.RootTools.physicsobjects.Photon import Photon

from CMGTools.TTHAnalysis.analyzers.ttHLepMCMatchAnalyzer import matchObjectCollection3
from CMGTools.RootTools.utils.DeltaR import deltaR, deltaPhi, bestMatch

class ttHPhotonAnalyzerSusy( Analyzer ):

    
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(ttHPhotonAnalyzerSusy,self).__init__(cfg_ana,cfg_comp,looperName)
        self.etaCentral = self.cfg_ana.etaCentral  if hasattr(self.cfg_ana, 'etaCentral') else 9999

    def declareHandles(self):
        super(ttHPhotonAnalyzerSusy, self).declareHandles()

    #----------------------------------------                                                                                                                                   
    # DECLARATION OF HANDLES OF PHOTONS STUFF                                                                                                                                   
    #----------------------------------------     

        #photons
        self.handles['photons'] = AutoHandle( self.cfg_ana.photons,'std::vector<pat::Photon>')

    def beginLoop(self):
        super(ttHPhotonAnalyzerSusy,self).beginLoop()
        self.counters.addCounter('events')
        count = self.counters.counter('events')
        count.register('all events')
        count.register('has >=1 gamma at preselection')
        count.register('has >=1 selected gamma')

##    tauID = "PhotonCutBasedID",

    def makePhotons(self, event):
        event.allphotons = map( Photon, self.handles['photons'].product() )
        event.allphotons.sort(key = lambda l : l.pt(), reverse = True)

        event.selectedPhotons = []
        event.selectedPhotonsCentral = []

        foundPhoton = False
        for gamma in event.allphotons:
            if gamma.pt() < self.cfg_ana.ptMin: continue
            if abs(gamma.eta()) > self.cfg_ana.etaMax: continue
            foundPhoton = True

            def idWP(gamma,X):
                """Create an integer equal to 1-2-3 for (loose,medium,tight)"""

## medium not stored
##                return gamma.photonID(X%"Loose") + gamma.photonID(X%"Medium") + gamma.photonID(X%"Tight")

                id=-1
                if gamma.photonID(X%"Loose"):
                    id=0
                if gamma.photonID(X%"Tight"):
                    id=2
                return id

            gamma.idCutBased = idWP(gamma, "PhotonCutBasedID%s")

            if gamma.photonID(self.cfg_ana.gammaID):
                event.selectedPhotons.append(gamma)
            
            if gamma.photonID(self.cfg_ana.gammaID) and abs(gamma.eta()) < self.etaCentral:
                event.selectedPhotonsCentral.append(gamma)

        event.selectedPhotons.sort(key = lambda l : l.pt(), reverse = True)
        event.selectedPhotonsCentral.sort(key = lambda l : l.pt(), reverse = True)

        self.counters.counter('events').inc('all events')
        if foundPhoton: self.counters.counter('events').inc('has >=1 gamma at preselection')
        if len(event.selectedPhotons): self.counters.counter('events').inc('has >=1 selected gamma')
       
    def matchPhotons(self, event):
        event.genPhotons = [ x for x in event.genParticles if x.status() == 3 and abs(x.pdgId()) == 22 ]
        match = matchObjectCollection3(event.allphotons, event.genPhotons, deltaRMax = 0.5)
        for gamma in event.allphotons:
            gen = match[gamma]
            gamma.mcMatchId = 1 if gen else 0

    def printInfo(self, event):
        print '----------------'
        if len(event.selectedPhotons)>0:
            print 'lenght: ',len(event.selectedPhotons)
            print 'gamma candidate pt: ',event.selectedPhotons[0].pt()
            print 'gamma candidate eta: ',event.selectedPhotons[0].eta()
            print 'gamma candidate phi: ',event.selectedPhotons[0].phi()
            print 'gamma candidate mass: ',event.selectedPhotons[0].mass()
            print 'gamma candidate HoE: ',event.selectedPhotons[0].hOVERe()
            print 'gamma candidate r9: ',event.selectedPhotons[0].r9()
            print 'gamma candidate sigmaIetaIeta: ',event.selectedPhotons[0].sigmaIetaIeta()
            print 'gamma candidate had iso: ',event.selectedPhotons[0].chargedHadronIso()
            print 'gamma candidate neu iso: ',event.selectedPhotons[0].neutralHadronIso()
            print 'gamma candidate gamma iso: ',event.selectedPhotons[0].photonIso()
            print 'gamma idCutBased',event.selectedPhotons[0].idCutBased


    def process(self, iEvent, event):
        self.readCollections( iEvent )
        #call the photons functions
        self.makePhotons(event)
        
#        self.printInfo(event)   

## ===> do matching                                                                                                                                                                                                     
        if not self.cfg_comp.isMC:
            return True

        self.matchPhotons(event)


        return True
