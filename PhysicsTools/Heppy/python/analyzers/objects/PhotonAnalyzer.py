import operator
import itertools
import copy
import types
import re

from ROOT import TLorentzVector
from ROOT import heppy 

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
        # footprint removed isolation
        self.doFootprintRemovedIsolation = getattr(cfg_ana, 'doFootprintRemovedIsolation', False)
        if self.doFootprintRemovedIsolation:
            self.footprintRemovedIsolationPUCorr =  self.cfg_ana.footprintRemovedIsolationPUCorr
            self.IsolationComputer = heppy.IsolationComputer()

    def declareHandles(self):
        super(PhotonAnalyzer, self).declareHandles()
        self.handles['rhoPhoton'] = AutoHandle( self.cfg_ana.rhoPhoton, 'double')

    #----------------------------------------                                                                                                                                   
    # DECLARATION OF HANDLES OF PHOTONS STUFF                                                                                                                                   
    #----------------------------------------     

        self.handles['photons'] = AutoHandle( self.cfg_ana.photons,'std::vector<pat::Photon>')
        self.mchandles['packedGen'] = AutoHandle( 'packedGenParticles', 'std::vector<pat::PackedGenParticle>' )
        self.mchandles['prunedGen'] = AutoHandle( 'prunedGenParticles', 'std::vector<reco::GenParticle>' )

        self.handles['packedCandidates'] = AutoHandle( 'packedPFCandidates', 'std::vector<pat::PackedCandidate>')
        self.handles['jets'] = AutoHandle( "slimmedJets", 'std::vector<pat::Jet>' )


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

        if self.doFootprintRemovedIsolation:
            # values are taken from EGamma implementation: https://github.com/cms-sw/cmssw/blob/CMSSW_7_6_X/RecoEgamma/PhotonIdentification/plugins/PhotonIDValueMapProducer.cc#L198-L199
            self.IsolationComputer.setPackedCandidates(self.handles['packedCandidates'].product(), -1, 0.1, 0.2)

        foundPhoton = False
        for gamma in event.allphotons:
            if gamma.pt() < self.cfg_ana.ptMin: continue
            if abs(gamma.eta()) > self.cfg_ana.etaMax: continue
            foundPhoton = True

            gamma.rho = float(self.handles['rhoPhoton'].product()[0])
            # https://twiki.cern.ch/twiki/bin/view/CMS/CutBasedPhotonIdentificationRun2#Selection_implementation_details
            if   abs(gamma.eta()) < 1.0:   gamma.EffectiveArea03 = [ 0.0234, 0.0053, 0.078  ]
            elif abs(gamma.eta()) < 1.479: gamma.EffectiveArea03 = [ 0.0189, 0.0103, 0.0629 ]
            elif abs(gamma.eta()) < 2.0:   gamma.EffectiveArea03 = [ 0.0171, 0.0057, 0.0264 ]
            elif abs(gamma.eta()) < 2.2:   gamma.EffectiveArea03 = [ 0.0129, 0.0070, 0.0462 ]
            elif abs(gamma.eta()) < 2.3:   gamma.EffectiveArea03 = [ 0.0110, 0.0152, 0.0740 ]
            elif abs(gamma.eta()) < 2.4:   gamma.EffectiveArea03 = [ 0.0074, 0.0232, 0.0924 ]
            else:                          gamma.EffectiveArea03 = [ 0.0035, 0.1709, 0.1484 ]

            if self.doFootprintRemovedIsolation:
                self.attachFootprintRemovedIsolation(gamma)

            gamma.relIso = (max(gamma.chargedHadronIso()-gamma.rho*gamma.EffectiveArea03[0],0) + max(gamma.neutralHadronIso()-gamma.rho*gamma.EffectiveArea03[1],0) + max(gamma.photonIso() - gamma.rho*gamma.EffectiveArea03[2],0))/gamma.pt()

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

            if self.cfg_ana.gammaID=="PhotonCutBasedIDLoose_CSA14" or self.cfg_ana.gammaID=="PhotonCutBasedIDLoose_PHYS14" :
                gamma.idCutBased = gamma.photonIDCSA14(self.cfg_ana.gammaID) 
                # we're keeing sigmaietaieta sidebands:
                keepThisPhoton   = gamma.photonIDCSA14(self.cfg_ana.gammaID, True) 

                if gamma.hasPixelSeed():
                    keepThisPhoton = False
                    gamma.idCutBased = 0
            elif "NoIso" in self.cfg_ana.gammaID:
                idName = re.split('_NoIso',self.cfg_ana.gammaID)
                keepThisPhoton = gamma.passPhotonID(idName[0],self.cfg_ana.conversionSafe_eleVeto)
                basenameID = re.split('_looseSieie',idName[0])
                gamma.idCutBased = gamma.passPhotonID(basenameID[0],self.cfg_ana.conversionSafe_eleVeto)
            else:
                # Reading from miniAOD directly
                #keepThisPhoton = gamma.photonID(self.cfg_ana.gammaID)

                # implement cut based ID with CMGTools
                keepThisPhoton = gamma.passPhotonID(self.cfg_ana.gammaID,self.cfg_ana.conversionSafe_eleVeto) and gamma.passPhotonIso(self.cfg_ana.gammaID,self.cfg_ana.gamma_isoCorr)

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
        event.genPhotonsMatched = [ x for x in event.genPhotonsWithMom if abs(x.mother(0).pdgId())<23 or x.mother(0).pdgId()==2212 ]
        match = matchObjectCollection3(event.allphotons, event.genPhotonsMatched, deltaRMax = 0.1)
        matchNoMom = matchObjectCollection3(event.allphotons, event.genPhotonsWithoutMom, deltaRMax = 0.1)
        packedGenParts = [ p for p in self.mchandles['packedGen'].product() if abs(p.eta()) < 3.1 ]
        partons = [ p for p in self.mchandles['prunedGen'].product() if (p.status()==23 or p.status()==22) and abs(p.pdgId())<22 ]
        for gamma in event.allphotons:
          gen = match[gamma]
          gamma.mcGamma = gen
          if gen and gen.pt()>=0.5*gamma.pt() and gen.pt()<=2.*gamma.pt():
            gamma.mcMatchId = 22
            sumPt03 = 0.;
            sumPt04 = 0.;
            for part in packedGenParts:
              if abs(part.pdgId())==12: continue # exclude neutrinos
              if abs(part.pdgId())==14: continue
              if abs(part.pdgId())==16: continue
              if abs(part.pdgId())==18: continue
              deltar = deltaR(gen.eta(), gen.phi(), part.eta(), part.phi())
              if deltar <= 0.3:
                sumPt03 += part.pt()
              if deltar <= 0.4:
                sumPt04 += part.pt()
            sumPt03 -= gen.pt()
            sumPt04 -= gen.pt()
            if sumPt03<0. : sumPt03=0.
            if sumPt04<0. : sumPt04=0.
            gamma.genIso03 = sumPt03
            gamma.genIso04 = sumPt04
            # match to parton
            deltaRmin = 999.
            for p in partons:
              deltar = deltaR(gen.eta(), gen.phi(), p.eta(), p.phi())
              if deltar < deltaRmin:
                deltaRmin = deltar
            gamma.drMinParton = deltaRmin
          else:
            genNoMom = matchNoMom[gamma]
            if genNoMom:
              gamma.mcMatchId = 7
              sumPt03 = 0.;
              sumPt04 = 0.;
              for part in packedGenParts:
                if abs(part.pdgId())==12: continue # exclude neutrinos
                if abs(part.pdgId())==14: continue
                if abs(part.pdgId())==16: continue
                if abs(part.pdgId())==18: continue
                deltar = deltaR(genNoMom.eta(), genNoMom.phi(), part.eta(), part.phi())
                if deltar <= 0.3:
                  sumPt03 += part.pt()
                if deltar <= 0.4:
                  sumPt04 += part.pt()
              sumPt03 -= genNoMom.pt()
              sumPt04 -= genNoMom.pt()
              if sumPt03<0. : sumPt03=0.
              if sumPt04<0. : sumPt04=0.
              gamma.genIso03 = sumPt03
              gamma.genIso04 = sumPt04
              # match to parton
              deltaRmin = 999.
              for p in partons:
                deltar = deltaR(genNoMom.eta(), genNoMom.phi(), p.eta(), p.phi())
                if deltar < deltaRmin:
                  deltaRmin = deltar
              gamma.drMinParton = deltaRmin
            else:
              gamma.mcMatchId = 0
              gamma.genIso03 = -1.
              gamma.genIso04 = -1.
              gamma.drMinParton = -1.





    def checkMatch( self, eta, phi, particles, deltar ):

      for part in particles:
        if deltaR(eta, phi, part.eta(), part.phi()) < deltar:
          return True

      return False





    def computeRandomCone( self, event, eta, phi, deltarmax, charged, jets, photons ):

      if self.checkMatch( eta, phi, jets, 2.*deltarmax ): 
        return -1.
    
      if self.checkMatch( eta, phi, photons, 2.*deltarmax ): 
        return -1.
    
      if self.checkMatch( eta, phi, event.selectedLeptons, deltarmax ): 
        return -1.

      iso = 0.

      for part in charged:
        if deltaR(eta, phi, part.eta(), part.phi()) > deltarmax : continue
        #if deltaR(eta, phi, part.eta(), part.phi()) < 0.02: continue
        iso += part.pt()

      return iso




            

    def randomCone( self, event ):

        patcands  = self.handles['packedCandidates'].product()
        jets  = self.handles['jets'].product()

        charged   = [ p for p in patcands if ( p.charge() != 0 and abs(p.pdgId())>20 and abs(p.dz())<=0.1 and p.fromPV()>1 and p.trackHighPurity() ) ]
        photons10 = [ p for p in patcands if ( p.pdgId() == 22 and p.pt()>10. ) ]
        jets20 = [ j for j in jets if j.pt() > 20 and abs(j.eta())<2.5 ]

        for gamma in event.allphotons:

          etaPhot = gamma.eta()
          phiPhot = gamma.eta()
          pi = 3.14159
          phiRC = phiPhot + 0.5*pi
          while phiRC>pi:
            phiRC -= 2.*pi


          gamma.chHadIsoRC03 = self.computeRandomCone( event, etaPhot, phiRC, 0.3, charged, jets20, photons10 )
          gamma.chHadIsoRC04 = self.computeRandomCone( event, etaPhot, phiRC, 0.4, charged, jets20, photons10 )
          
          
          #try other side
          phiRC = phiPhot - 0.5*pi
          while phiRC<-pi:
            phiRC += 2.*pi
          
          if gamma.chHadIsoRC03<0. : gamma.chHadIsoRC03 = self.computeRandomCone( event, etaPhot, phiRC, 0.3, charged, jets20, photons10 )
          if gamma.chHadIsoRC04<0. : gamma.chHadIsoRC04 = self.computeRandomCone( event, etaPhot, phiRC, 0.4, charged, jets20, photons10 )


    def attachFootprintRemovedIsolation(self, gamma):
        # cone deltar=0.3
        gamma.ftprAbsIsoCharged03 = self.IsolationComputer.chargedAbsIso(gamma.physObj, 0.3, 0, 0.0);
        gamma.ftprAbsIsoPho03     = self.IsolationComputer.photonAbsIsoRaw( gamma.physObj, 0.3, 0, 0.0);
        gamma.ftprAbsIsoNHad03    = self.IsolationComputer.neutralHadAbsIsoRaw( gamma.physObj, 0.3, 0, 0.0);
        gamma.ftprAbsIso03 = gamma.ftprAbsIsoCharged03 + gamma.ftprAbsIsoPho03 + gamma.ftprAbsIsoNHad03
        if self.cfg_ana.gamma_isoCorr == "rhoArea":
            gamma.ftprAbsIso03 = (max(gamma.ftprAbsIsoCharged03-gamma.rho*gamma.EffectiveArea03[0],0) + max(gamma.ftprAbsIsoPho03-gamma.rho*gamma.EffectiveArea03[1],0) + max(gamma.ftprAbsIsoNHad03 - gamma.rho*gamma.EffectiveArea03[2],0))
        elif self.cfg_ana.gamma_isoCorr != 'raw':
            raise RuntimeError, "Unsupported gamma_isoCorr name '" + str(self.cfg_ana.gamma_isoCorr) +  "'! For now only 'rhoArea', 'raw' are supported."
        gamma.ftprRelIso03 = gamma.ftprAbsIso03/gamma.pt()

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

        if self.cfg_ana.do_randomCone:
            self.randomCone(event)

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
    gammaID = "PhotonCutBasedIDLoose_CSA14",
    rhoPhoton = 'fixedGridRhoFastjetAll',
    gamma_isoCorr = 'rhoArea',
    # Footprint-removed isolation, removing all the footprint of the photon
    doFootprintRemovedIsolation = False, # off by default since it requires access to all PFCandidates
    packedCandidates = 'packedPFCandidates',
    footprintRemovedIsolationPUCorr = 'rhoArea', # Allowed options: 'rhoArea', 'raw' (uncorrected)
    conversionSafe_eleVeto = False,
    do_mc_match = True,
    do_randomCone = False,
  )
)

