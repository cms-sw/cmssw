from CMGTools.H2TauTau.proto.analyzers.H2TauTauTreeProducer import H2TauTauTreeProducer


class H2TauTauTreeProducerMuMu(H2TauTauTreeProducer):

    '''Tree producer for the H->tau tau->mu mu analysis.'''

    def declareVariables(self, setup):

        super(H2TauTauTreeProducerMuMu, self).declareVariables(setup)

        self.bookMuon(self.tree, 'l1')
        self.bookMuon(self.tree, 'l2')

        self.bookGenParticle(self.tree, 'l1_gen')
        self.bookGenParticle(self.tree, 'l2_gen')

    def process(self, event):
        super(H2TauTauTreeProducerMuMu, self).process(event)

        mu1 = event.diLepton.leg1()
        mu2 = event.diLepton.leg2()

        self.fillMuon(self.tree, 'l1', mu1)
        self.fillMuon(self.tree, 'l2', mu2)

        if hasattr(mu1, 'genp'):
            self.fillGenParticle(self.tree, 'l1_gen', mu1.genp)
        if hasattr(mu2, 'genp'):
            self.fillGenParticle(self.tree, 'l2_gen', mu2.genp)

        self.fillTree(event)
        