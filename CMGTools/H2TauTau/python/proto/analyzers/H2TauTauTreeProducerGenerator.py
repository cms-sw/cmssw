from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from PhysicsTools.Heppy.physicsobjects import GenParticle
from PhysicsTools.Heppy.physicsobjects.PhysicsObjects import GenJet
from PhysicsTools.HeppyCore.utils.deltar import cleanObjectCollection

from PhysicsTools.Heppy.analyzers.core.TreeAnalyzerNumpy import TreeAnalyzerNumpy
from CMGTools.H2TauTau.proto.analyzers.ntuple import *

class H2TauTauTreeProducerGenerator( TreeAnalyzerNumpy ):
    '''Tree producer for the H->tau tau analysis.'''
    
    def getAllDaughters(self, p, l):
        for i in range(0, p.numberOfDaughters()):
            d = p.daughter(i)
            l.append(d)
            self.getAllDaughters(d, l)

    # for pythia 8
    def finalSelf(self, p):
        if p.numberOfDaughters() == 1 and p.daughter(0).pdgId() == p.pdgId():
            return self.finalSelf(p.daughter(0))
        # check for radiated photons
        if p.numberOfDaughters() == 2:
            # FIXME - more elegant solution?
            if p.daughter(0).pdgId() == p.pdgId() and p.daughter(1).pdgId() == 22:
                return self.finalSelf(p.daughter(0))
            if p.daughter(1).pdgId() == p.pdgId() and p.daughter(0).pdgId() == 22:
                return self.finalSelf(p.daughter(1))

        return p

    def isFinal(self, p):
        if p.numberOfDaughters() > 0:
            for i_d in range(len(p.numberOfDaughters())):
                if p.daughter(i_d).pdgId() == p.pdgId():
                    return False
        return True

    def mother(self, p):
        if p.numberOfMothers() == 1:
            if p.mother(0).pdgId() != p.pdgId():
                return p.mother(0)
            else:
                return self.mother(p.mother(0))
        return None

    def getVisibleP4(self, gen):
        p4vis = gen.p4()
        daughters = []
        self.getAllDaughters(gen, daughters)
        invisDaughters = [p for p in daughters if abs(p.pdgId()) in [12, 14, 16]]
        for d in invisDaughters:
            p4vis -= d.p4()

        return p4vis

    def findCentralJets( self, leadJets, otherJets ):
        '''Finds all jets between the 2 leading jets, for central jet veto.'''
        if not len(otherJets):
            return []
        etamin = leadJets[0].eta()
        etamax = leadJets[1].eta()
        if etamin > etamax:
            etamin, etamax = etamax, etamin
        def isCentral( jet ):
            if jet.pt() < 30.:
                return False
            eta = jet.eta()
            if etamin < eta and eta < etamax:
                return True
            else:
                return False
        centralJets = filter( isCentral, otherJets )
        return centralJets


    def declareVariables(self):

        tr = self.tree

        var( tr, 'run', int)
        var( tr, 'lumi', int)
        var( tr, 'evt', int)

        bookGenParticle(tr, 'higgs')
        bookGenParticle(tr, 'l1_gen')
        bookGenParticle(tr, 'l2_gen')
        bookGenParticle(tr, 'genjet1')
        bookGenParticle(tr, 'genjet2')

        var(tr, 'mjj')
        var(tr, 'deta')
        var(tr, 'nCentral')
        var(tr, 'nGenJets')

        var(tr, 'l1_gen_vis_pt')
        var(tr, 'l1_gen_vis_eta')
        var(tr, 'l1_gen_vis_phi')
        var(tr, 'l1_gen_vis_m')
        var(tr, 'l1_gen_decay_pdgId')

        var(tr, 'l2_gen_vis_pt')
        var(tr, 'l2_gen_vis_eta')
        var(tr, 'l2_gen_vis_phi')
        var(tr, 'l2_gen_vis_m')
        var(tr, 'l2_gen_decay_pdgId')
        
        var(tr, 'genMet')
        var(tr, 'genMex')
        var(tr, 'genMey')

        self.maxNGenJets = 4
        for i in range(0, self.maxNGenJets):
            bookGenParticle(tr, 'genQG_{i}'.format(i=i))


         
    def declareHandles(self):
        super(H2TauTauTreeProducerGenerator, self).declareHandles()


        self.src = 'genParticlesPruned'
        self.gensrc = 'slimmedGenJets'
        self.packedsrc = 'packedGenParticles'


        if hasattr( self.cfg_ana, 'src'):
            self.src = self.cfg_ana.src

        if hasattr( self.cfg_ana, 'gensrc'):
            self.gensrc = self.cfg_ana.gensrc

        if hasattr( self.cfg_ana, 'packedsrc'):
            self.packedsrc = self.cfg_ana.packedsrc


        self.mchandles['genParticles'] = AutoHandle( self.src,
                                                     'std::vector<reco::GenParticle>' )

        self.mchandles['genJets'] = AutoHandle(self.gensrc,
                                               'std::vector<reco::GenJet>')

        self.mchandles['packedgenParticles'] = AutoHandle(self.packedsrc,
                                                          'std::vector<pat::PackedGenParticle>')

        
        
    def process(self, event):
        self.readCollections( event.input )
                
        tr = self.tree
        tr.reset()

        fill( tr, 'run', event.input.eventAuxiliary().id().run())
        fill( tr, 'lumi', event.input.eventAuxiliary().id().luminosityBlock())
        fill( tr, 'evt', event.input.eventAuxiliary().id().event())

        genParticles = self.mchandles['genParticles'].product()
        event.genParticles = map( GenParticle, genParticles)

#         # For Z
# #        parent = 23
# #        parent_status = 22

#         # For Higgs
#         parent = 25
#         parent_status = 3

        bosonParents = [23, 25]

        bosonStati = [3, 22]

        bosons = [self.finalSelf(gen) for gen in event.genParticles if gen.status() in bosonStati and abs(gen.pdgId()) in bosonParents]

        if len(bosons)!=1:
            strerr = '{nhiggs} bosons! :\n {comp}'.format(nhiggs=len(higbosonsgsBosons), comp=str(self.cfg_comp))
            print strerr
            return False
        
        boson = bosons[0]

        # Get generated taus
        taus = []
        for i_d in range(boson.numberOfDaughters()):
            daughter = boson.daughter(i_d)
            if abs(daughter.pdgId()) == 15:
                taus.append(self.finalSelf(daughter))

        if len(taus) != 2:
            print 'Non-tau decay'
            return False

        # Get tau daughters
        tauDecayIds = []
        muon = None
        had_tau = None
        for tau in taus:
            decayId = 15
            for i_d in range(tau.numberOfDaughters()):
                if abs(tau.daughter(i_d).pdgId()) == 13:
                    muon = self.finalSelf(tau.daughter(i_d))
                    decayId = 13
                    continue
                elif abs(tau.daughter(i_d).pdgId()) == 11:
                    decayId = 11
                    continue
            if decayId == 15:
                had_tau = tau
            tauDecayIds.append(decayId)

        if not (13 in tauDecayIds and 15 in tauDecayIds):
            print 'Not a mu-tau decay'
            return False


        fillGenParticle(tr, 'l2_gen', muon)
        genVisTau = self.getVisibleP4(muon)
        fill(tr, 'l2_gen_vis_pt', genVisTau.pt())
        fill(tr, 'l2_gen_vis_eta', genVisTau.eta())
        fill(tr, 'l2_gen_vis_phi', genVisTau.phi())
        fill(tr, 'l2_gen_vis_m', genVisTau.mass())
        fill(tr, 'l2_gen_decay_pdgId', muon.pdgId())



        fillGenParticle(tr, 'higgs', boson)


        fillGenParticle(tr, 'l1_gen', had_tau)
        genVisTau1 = self.getVisibleP4(had_tau)
        fill(tr, 'l1_gen_vis_pt', genVisTau1.pt())
        fill(tr, 'l1_gen_vis_eta', genVisTau1.eta())
        fill(tr, 'l1_gen_vis_phi', genVisTau1.phi())
        fill(tr, 'l1_gen_vis_m', genVisTau1.mass())
        fill(tr, 'l1_gen_decay_pdgId', had_tau.pdgId())
        
        daughters = [muon, had_tau]

        genJets = self.mchandles['genJets'].product()
        event.genJets = map(GenJet, genJets)

        selGenJets = [jet for jet in event.genJets if jet.pt() > 30 and abs(jet.eta()) < 4.7]
        
        cleanGenJets, dummy = cleanObjectCollection( selGenJets,
                                                     masks = daughters,
                                                     deltaRMin = 0.5)

        cleanGenJets.sort(key=lambda x: -x.pt())

        deta = -1
        mjj = -1
        ncentral = -1
        
        fill(tr, 'nGenJets', len(cleanGenJets))

        if len(cleanGenJets) >= 2:
            deta = cleanGenJets[0].eta() - cleanGenJets[1].eta()
            dijetp4 = cleanGenJets[0].p4() + cleanGenJets[1].p4()
            mjj = dijetp4.M()
            
            leadJets = cleanGenJets[:2]
            otherJets = cleanGenJets[2:]
            centralJets = self.findCentralJets( leadJets, otherJets )
            ncentral = len(centralJets)
            
            fill(tr, 'mjj', mjj)
            fill(tr, 'deta', deta)
            fill(tr, 'nCentral', ncentral)
            
            

        if len(cleanGenJets)>=1:
            fillGenParticle(tr, 'genjet1', cleanGenJets[0])
        if len(cleanGenJets)>=2:
            fillGenParticle(tr, 'genjet2', cleanGenJets[1])


        neutrinos = [p for p in event.genParticles if abs(p.pdgId()) in (12, 14, 16) and self.isFinal(p)]
        if neutrinos:
            genMet = neutrinos[0].p4()
            for p in neutrinos[1:]:
                genMet += p.p4()
            fill(tr, 'genMet', p.pt())
            fill(tr, 'genMex', p.px())
            fill(tr, 'genMey', p.py())
        else:
            fill(tr, 'genMet', 0.)
            fill(tr, 'genMex', 0.)
            fill(tr, 'genMey', 0.)
          

        quarksGluons = [p for p in event.genParticles if abs(p.pdgId()) in (1, 2, 3, 4, 5, 21) and p.status() == 3 and (p.numberOfDaughters() == 0 or p.daughter(0).status() != 3)]
        quarksGluons.sort(key=lambda x: -x.pt())
        for i in range(0, min(self.maxNGenJets, len(quarksGluons))):
            fillGenParticle(tr, 'genQG_{i}'.format(i=i), quarksGluons[i])
            
            
        self.tree.tree.Fill()

