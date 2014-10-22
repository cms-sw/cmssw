from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from PhysicsTools.Heppy.physicsobjects.Jet import Jet, GenJet
from PhysicsTools.HeppyCore.utils.deltar import matchObjectCollection

class SimpleJetAnalyzer(Analyzer):
    '''Just a simple jet analyzer, to be used in tutorials.'''

    def declareHandles(self):
        super(SimpleJetAnalyzer, self).declareHandles()
        self.handles['jets'] = AutoHandle( 'slimmedJets',
                                           'std::vector<pat::Jet>' )
        self.mchandles['genjets'] = AutoHandle( 'slimmedGenJets',
                                                'std::vector<reco::GenJet>')
        
    def process(self, event):
        super(SimpleJetAnalyzer, self).readCollections(event.input)
        event.jets = map(Jet, self.handles['jets'].product())
        event.jets = [ jet for jet in event.jets if jet.pt()>self.cfg_ana.ptmin]
        # print 'event', event.iEv
        # for jet in event.jets:
        #    print jet

        if self.cfg_comp.isMC:
            event.genjets =  map(GenJet, self.mchandles['genjets'].product())
            matches = matchObjectCollection(event.jets, event.genjets, 0.2)
            for jet in event.jets:
                jet.gen = matches[jet]
            
