import operator
import itertools
import copy
import types

from ROOT import TLorentzVector

from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.HeppyCore.framework.event import Event
from PhysicsTools.HeppyCore.statistics.counter import Counter, Counters
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from PhysicsTools.Heppy.physicsobjects.Photon import Photon

from PhysicsTools.HeppyCore.utils.deltar import deltaR, deltaPhi, bestMatch, matchObjectCollection3

import PhysicsTools.HeppyCore.framework.config as cfg


class PhotonAnalyzer( Analyzer ):

    
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(PhotonAnalyzer,self).__init__(cfg_ana,cfg_comp,looperName)
        self.etaCentral = self.cfg_ana.etaCentral  if hasattr(self.cfg_ana, 'etaCentral') else 9999

    def declareHandles(self):
        super(PhotonAnalyzer, self).declareHandles()

    #----------------------------------------                                                                                                                                   
    # DECLARATION OF HANDLES OF PHOTONS STUFF                                                                                                                                   
    #----------------------------------------     

        #photons
        self.handles['photons'] = AutoHandle( self.cfg_ana.photons,'std::vector<pat::Photon>')
        self.mchandles['packedGen'] = AutoHandle( 'packedGenParticles', 'std::vector<pat::PackedGenParticle>' )

    def beginLoop(self, setup):
        super(PhotonAnalyzer,self).beginLoop(setup)
        self.counters.addCounter('events')
        count = self.counters.counter('events')
        count.register('all events')
        count.register('has >=1 gamma at preselection')
        count.register('has >=1 selected gamma')

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

                id=0
                if gamma.photonID(X%"Loose"):
                    id=1
                #if gamma.photonID(X%"Medium"):
                #    id=2 
                if gamma.photonID(X%"Tight"):
                    id=3
                return id

            gamma.idCutBased = idWP(gamma, "PhotonCutBasedID%s")


            keepThisPhoton = True
            if self.cfg_ana.gammaID=="PhotonCutBasedIDLoose_CSA14" :
                keepThisPhoton = gamma.photonIDCSA14("PhotonCutBasedIDLoose_CSA14")
                gamma.idCutBased = keepThisPhoton
                # we're keeing sigmaietaieta sidebands, but the id is false for them:
                
                if abs(gamma.eta())< 1.479 and gamma.full5x5_sigmaIetaIeta()>0.010 : 
                    gamma.idCutBased = False
                if abs(gamma.eta())>=1.479 and gamma.full5x5_sigmaIetaIeta()>0.0321 : 
                    gamma.idCutBased = False
            else:
              keepThisPhoton = gamma.photonID(self.cfg_ana.gammaID)

            if gamma.hasPixelSeed():
              keepThisPhoton = False
              gamma.idCutBased = 0


            if keepThisPhoton:
                event.selectedPhotons.append(gamma)

            if keepThisPhoton and abs(gamma.eta()) < self.etaCentral:
                event.selectedPhotonsCentral.append(gamma)

        event.selectedPhotons.sort(key = lambda l : l.pt(), reverse = True)
        event.selectedPhotonsCentral.sort(key = lambda l : l.pt(), reverse = True)

        self.counters.counter('events').inc('all events')
        if foundPhoton: self.counters.counter('events').inc('has >=1 gamma at preselection')
        if len(event.selectedPhotons): self.counters.counter('events').inc('has >=1 selected gamma')
       
    def matchPhotons(self, event):
        event.genPhotons = [ x for x in event.genParticles if x.status() == 1 and abs(x.pdgId()) == 22 ]
        event.genPhotonsWithMom = [ x for x in event.genPhotons if x.numberOfMothers()>0 ]
        event.genPhotonsWithoutMom = [ x for x in event.genPhotons if x.numberOfMothers()==0 ]
        event.genPhotonsMatched = [ x for x in event.genPhotonsWithMom if abs(x.mother(0).pdgId())<23 ]
        match = matchObjectCollection3(event.allphotons, event.genPhotonsMatched, deltaRMax = 0.1)
        matchNoMom = matchObjectCollection3(event.allphotons, event.genPhotonsWithoutMom, deltaRMax = 0.1)
        packedGenParts = [ p for p in self.mchandles['packedGen'].product() if abs(p.eta()) < 3.1 ]
        for gamma in event.allphotons:
          gen = match[gamma]
          if gen and gen.pt()>=0.5*gamma.pt() and gen.pt()<=2.*gamma.pt():
            gamma.mcMatchId = 22
            sumPt = 0.;
            for part in packedGenParts:
              if abs(part.pdgId())==12: continue # exclude neutrinos
              if abs(part.pdgId())==14: continue
              if abs(part.pdgId())==16: continue
              if abs(part.pdgId())==18: continue
              if deltaR(gen.eta(), gen.phi(), part.eta(), part.phi()) > 0.4: continue
              sumPt += part.pt()
            sumPt -= gen.pt()
            if sumPt<0. : sumPt=0.
            gamma.genIso = sumPt
          else:
            genNoMom = matchNoMom[gamma]
            if genNoMom:
              gamma.mcMatchId = 7
              sumPt = 0.;
              for part in packedGenParts:
                if abs(part.pdgId())==12: continue # exclude neutrinos
                if abs(part.pdgId())==14: continue
                if abs(part.pdgId())==16: continue
                if abs(part.pdgId())==18: continue
                if deltaR(genNoMom.eta(), genNoMom.phi(), part.eta(), part.phi()) > 0.4: continue
                sumPt += part.pt()
              sumPt -= genNoMom.pt()
              if sumPt<0. : sumPt=0.
              gamma.genIso = sumPt
            else:
              gamma.mcMatchId = 0
              gamma.genIso = -1.

    def printInfo(self, event):
        print '----------------'
        if len(event.selectedPhotons)>0:
            print 'lenght: ',len(event.selectedPhotons)
            print 'gamma candidate pt: ',event.selectedPhotons[0].pt()
            print 'gamma candidate eta: ',event.selectedPhotons[0].eta()
            print 'gamma candidate phi: ',event.selectedPhotons[0].phi()
            print 'gamma candidate mass: ',event.selectedPhotons[0].mass()
            print 'gamma candidate HoE: ',event.selectedPhotons[0].hOVERe()
            print 'gamma candidate r9: ',event.selectedPhotons[0].full5x5_r9()
            print 'gamma candidate sigmaIetaIeta: ',event.selectedPhotons[0].full5x5_sigmaIetaIeta()
            print 'gamma candidate had iso: ',event.selectedPhotons[0].chargedHadronIso()
            print 'gamma candidate neu iso: ',event.selectedPhotons[0].neutralHadronIso()
            print 'gamma candidate gamma iso: ',event.selectedPhotons[0].photonIso()
            print 'gamma idCutBased',event.selectedPhotons[0].idCutBased


    def process(self, event):
        self.readCollections( event.input )
        self.makePhotons(event)
#        self.printInfo(event)   

        if not self.cfg_comp.isMC:
            return True

        if self.cfg_ana.do_mc_match and hasattr(event, 'genParticles'):
            self.matchPhotons(event)

        return True


setattr(PhotonAnalyzer,"defaultConfig",cfg.Analyzer(
    class_object=PhotonAnalyzer,
    photons='slimmedPhotons',
    ptMin = 20,
    etaMax = 2.5,
    gammaID = "PhotonCutBasedIDLoose",
    do_mc_match = True,
  )
)
