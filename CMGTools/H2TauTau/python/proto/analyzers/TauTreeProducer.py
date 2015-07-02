from PhysicsTools.Heppy.physicsutils.TauDecayModes import tauDecayModes

from CMGTools.H2TauTau.proto.analyzers.H2TauTauTreeProducerBase import H2TauTauTreeProducerBase

class TauTreeProducer(H2TauTauTreeProducerBase):
    ''' Tree producer for tau POG study.
    '''

    def __init__(self, *args):
        super(TauTreeProducer, self).__init__(*args)
        self.maxNTaus = 5

    def declareHandles(self):
        super(TauTreeProducer, self).declareHandles()

    def declareVariables(self, setup):

        self.bookTau(self.tree, 'tau')
        self.bookGenParticle(self.tree, 'tau_gen')
        self.bookGenParticle(self.tree, 'tau_gen_vis')
        self.var(self.tree, 'tau_gen_decayMode')


    def process(self, event):
        # needed when doing handle.product(), goes back to
        # PhysicsTools.Heppy.analyzers.core.Analyzer
        self.readCollections(event.input)


        if not eval(self.skimFunction):
            return False

        for i_tau, tau in enumerate(event.selectedTaus):
            
            if i_tau < self.maxNTaus:
                self.tree.reset()
                self.fillTau(self.tree, 'tau', tau)
                if tau.mcTau:
                    self.fillGenParticle(self.tree, 'tau_gen', tau.mcTau)
                    if tau.genJet():
                        self.fillGenParticle(self.tree, 'tau_gen_vis', tau.genJet())
                        self.fill(self.tree, 'tau_gen_decayMode', tauDecayModes.genDecayModeInt(tau.genJet()))

                self.fillTree(event)
