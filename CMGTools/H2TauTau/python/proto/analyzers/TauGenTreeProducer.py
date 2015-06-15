import ROOT

from PhysicsTools.Heppy.physicsutils.TauDecayModes import tauDecayModes

from CMGTools.H2TauTau.proto.analyzers.H2TauTauTreeProducerBase import H2TauTauTreeProducerBase

class TauGenTreeProducer(H2TauTauTreeProducerBase):
    ''' Tree producer for generator tau study.
    '''

    def __init__(self, *args):
        super(TauGenTreeProducer, self).__init__(*args)

    def declareHandles(self):
        super(TauGenTreeProducer, self).declareHandles()

    @staticmethod
    def finalDaughters(gen, daughters=None):
        if daughters is None:
            daughters = []
        for i in range(gen.numberOfDaughters()):
            daughter = gen.daughter(i)
            if daughter.numberOfDaughters() == 0:
                daughters.append(daughter)
            else:
                TauGenTreeProducer.finalDaughters(daughter, daughters)

        return daughters

    @staticmethod
    def visibleP4(gen):
        final_ds = TauGenTreeProducer.finalDaughters(gen)

        p4 = sum((d.p4() for d in final_ds if abs(d.pdgId()) not in [12, 14, 16]), ROOT.math.XYZTLorentzVectorD())

        return p4

    def declareVariables(self, setup):

        self.bookTau(self.tree, 'tau1')
        self.bookGenParticle(self.tree, 'tau1_gen')
        self.bookParticle(self.tree, 'tau1_gen_vis')
        self.var(self.tree, 'tau1_gen_decayMode')

        self.bookTau(self.tree, 'tau2')
        self.bookGenParticle(self.tree, 'tau2_gen')
        self.bookParticle(self.tree, 'tau2_gen_vis')
        self.var(self.tree, 'tau2_gen_decayMode')

        self.var(self.tree, 'n_gen_taus')
        self.var(self.tree, 'n_gen_tauleps')


    def process(self, event):
        # needed when doing handle.product(), goes back to
        # PhysicsTools.Heppy.analyzers.core.Analyzer
        self.readCollections(event.input)


        if not eval(self.skimFunction):
            return False

        self.tree.reset()
        

        n_gen_tau = 0
        for gen_tau in event.gentaus:
            # FIXME - temporary, let's see for a longer-term solution...
            if abs(gen_tau.mother().pdgId()) in [23, 24]:
                continue

            if n_gen_tau >= 2:
                print 'More than two generated hadronic taus!'
                continue

            self.fillGenParticle(self.tree, 'tau{i}_gen'.format(i=n_gen_tau+1), gen_tau)
            self.fillParticle(self.tree, 'tau{i}_gen_vis'.format(i=n_gen_tau+1), self.visibleP4(gen_tau))
            self.fill(self.tree, 'tau{i}_gen_decayMode'.format(i=n_gen_tau+1), tauDecayModes.genDecayModeInt([d for d in TauGenTreeProducer.finalDaughters(gen_tau) if abs(d.pdgId()) not in [12, 14, 16]]))

            for tau in event.selectedTaus:
                if tau.mcTau == gen_tau:
                    # import pdb; pdb.set_trace()
                    self.fillTau(self.tree, 'tau{i}'.format(i=n_gen_tau+1), tau)
                    # if tau.genJet():
                        # self.fillGenParticle(self.tree, 'tau{i}_gen_vis'.format(i=n_gen_tau), tau.genJet())

            n_gen_tau += 1

        self.fill(self.tree, 'n_gen_taus', n_gen_tau)
                            
        n_gen_taulep = 0
        for gen_tau_lep in event.gentauleps:
            tau = gen_tau_lep.mother()
            if abs(tau.pdgId()) != 15:
                continue
            if abs(tau.mother().pdgId()) == 15:
                tau = tau.mother()
            if abs(tau.mother().pdgId()) in [23, 24]:
                continue
            
            n_gen_taulep += 1

        
        self.fill(self.tree, 'n_gen_tauleps', n_gen_taulep)

        self.fillTree(event)
