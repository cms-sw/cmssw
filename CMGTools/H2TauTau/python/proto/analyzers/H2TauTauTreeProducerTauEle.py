from CMGTools.H2TauTau.proto.analyzers.H2TauTauTreeProducer import H2TauTauTreeProducer

class H2TauTauTreeProducerTauEle(H2TauTauTreeProducer):
    '''Tree producer for the H->tau tau analysis.'''

    def declareVariables(self, setup):

        super(H2TauTauTreeProducerTauEle, self).declareVariables(setup)

        self.bookTau(self.tree, 'l1')
        self.bookEle(self.tree, 'l2')

        self.bookGenParticle(self.tree, 'l1_gen')
        self.var(self.tree, 'l1_gen_lepfromtau', int)
        self.bookGenParticle(self.tree, 'l2_gen')
        self.var(self.tree, 'l2_gen_lepfromtau', int)

        self.bookParticle(self.tree, 'l1_gen_vis')

        self.var(self.tree, 'l1_weight_fakerate')
        self.var(self.tree, 'l1_weight_fakerate_up')
        self.var(self.tree, 'l1_weight_fakerate_down')

        self.var( self.tree, 'weight_zll')

    def process(self, event):

        super(H2TauTauTreeProducerTauEle, self).process(event)

        tau = event.diLepton.leg1()
        ele = event.diLepton.leg2()

        self.fillTau(self.tree, 'l1', tau)
        self.fillEle(self.tree, 'l2', ele)

        if tau.genp:
            self.fillGenParticle(self.tree, 'l1_gen', tau.genp)
            self.fill(self.tree, 'l1_gen_lepfromtau', tau.isTauLep)
        if ele.genp:
            self.fillGenParticle(self.tree, 'l2_gen', ele.genp)
            self.fill(self.tree, 'l2_gen_lepfromtau', ele.isTauLep)
        
        # save the p4 of the visible tau products at the generator level
        if tau.genJet() and hasattr(tau, 'genp') and abs(tau.genp.pdgId()) == 15:
            self.fillParticle(self.tree, 'l1_gen_vis', tau.physObj.genJet())

        self.fill(self.tree, 'l1_weight_fakerate', event.tauFakeRateWeightUp)
        self.fill(self.tree, 'l1_weight_fakerate_up', event.tauFakeRateWeightDown)
        self.fill(self.tree, 'l1_weight_fakerate_down', event.tauFakeRateWeight)

        self.fill(self.tree, 'weight_zll', event.zllWeight)

        self.fillTree(event)
