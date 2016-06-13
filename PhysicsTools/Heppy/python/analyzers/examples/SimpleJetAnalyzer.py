from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from PhysicsTools.Heppy.physicsobjects.Jet import Jet, GenJet
from PhysicsTools.HeppyCore.utils.deltar import matchObjectCollection

class SimpleJetAnalyzer(Analyzer):
    '''Just a simple jet analyzer, to be used in tutorials.

    example configuration:
    
    jets = cfg.Analyzer(
       SimpleJetAnalyzer,
       'jets',
       filter_func = lambda x : x.pt()>30 # filtering function for the jets
       njets = 4, # keeping the first 4 leading jets passing cuts 
    )
    '''

    def declareHandles(self):
        super(SimpleJetAnalyzer, self).declareHandles()
        self.handles['jets'] = AutoHandle( 'slimmedJets',
                                           'std::vector<pat::Jet>' )
        self.mchandles['genjets'] = AutoHandle( 'slimmedGenJets',
                                                'std::vector<reco::GenJet>')
        
    def process(self, event):
        super(SimpleJetAnalyzer, self).readCollections(event.input)
        # creating Jet python objects wrapping the EDM jets
        jets = map(Jet, self.handles['jets'].product())
        jets = [ jet for jet in jets if self.cfg_ana.filter_func(jet)]
        jets = jets[:self.cfg_ana.njets]

        if self.cfg_comp.isMC:
            genjets =  map(GenJet, self.mchandles['genjets'].product())
            matches = matchObjectCollection(jets, genjets, 0.2)
            for jet in jets:
                jet.gen = matches[jet]
            
        setattr(event, self.instance_label, jets)
        setattr(event, '_'.join([self.instance_label, 'gen']), genjets)
