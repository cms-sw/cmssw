from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle

from CMGTools.H2TauTau.proto.analyzers.H2TauTauTreeProducerBase import H2TauTauTreeProducerBase

class H2TauTauTreeProducer(H2TauTauTreeProducerBase):

    '''
       Base H->tautau tree producer.
       Books and fills the branches that are common to
       all four channels (mt, et, tt, em).
       The following branches are booked and filled:
       _ event ID
       _ di-tau pair variables (including MET)
       _ raw pf MET
       _ pass_leptons (both legs pass tight ID and isolation)
       _ third lepton veto
       _ dilepton veto
       _ VBF variables
       _ generator information variables (including NUP)
       _ jet1 and jet2 variables (sorted by pt)
       _ bjet1 and bjet2 variables (sorted by pt)
       _ event weight
       _ vertex weight
       _ embed weight
       _ gen parent boson H, W, Z (if exists)
       _ weight_njet
       _ event rho
       _ HqT weights
       Signal lepton-specific variables need to be booked
       and filled in the channel-specific child producers.

       The branch names can be changed by means of a dictionary.
    '''

    def __init__(self, *args):
        super(H2TauTauTreeProducer, self).__init__(*args)

    def declareHandles(self):
        super(H2TauTauTreeProducer, self).declareHandles()
        self.handles['pfmetraw'] = AutoHandle(
            'slimmedMETs',
            'std::vector<pat::MET>'
        )

    def declareVariables(self, setup):

        self.bookEvent(self.tree)
        self.bookDiLepton(self.tree)
        self.bookGenInfo(self.tree)
        self.bookVBF(self.tree, 'vbf')

        self.bookJet(self.tree, 'jet1')
        self.bookJet(self.tree, 'jet2')

        self.bookJet(self.tree, 'bjet1')
        self.bookJet(self.tree, 'bjet2')

        self.bookParticle(self.tree, 'pfmet')

        self.bookGenParticle(self.tree, 'genboson')


    def process(self, event):

        # needed when doing handle.product(), goes back to
        # PhysicsTools.Heppy.analyzers.core.Analyzer
        self.readCollections(event.input)

        self.tree.reset()

        if not eval(self.skimFunction):
            return False

        self.fillEvent(self.tree, event)
        self.fillDiLepton(self.tree, event.diLepton)
        self.fillGenInfo(self.tree, event)
        if hasattr(event, 'vbf'):
            self.fillVBF(self.tree, 'vbf', event.vbf)

        for i, jet in enumerate(event.cleanJets[:2]):
            self.fillJet(self.tree, 'jet{n}'.format(n=str(i + 1)), jet)

        for i, jet in enumerate(event.cleanBJets[:2]):
            self.fillJet(self.tree, 'bjet{n}'.format(n=str(i + 1)), jet)

        if hasattr(event, 'parentBoson'):
            self.fillGenParticle(self.tree, 'genboson', event.parentBoson)

        pfmet = self.handles['pfmetraw'].product()[0]
        self.fillParticle(self.tree, 'pfmet', pfmet)

        if type(self) is H2TauTauTreeProducer:
            self.fillTree(event)
