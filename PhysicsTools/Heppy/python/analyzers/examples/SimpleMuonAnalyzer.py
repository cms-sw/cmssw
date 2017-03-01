from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from PhysicsTools.Heppy.physicsobjects.Muon import Muon


class SimpleMuonAnalyzer(Analyzer):
    '''Just a simple jet analyzer, to be used in tutorials.'''

    def declareHandles(self):
        super(SimpleMuonAnalyzer, self).declareHandles()
        self.handles['muons'] = AutoHandle( 'slimmedMuons',
                                            'std::vector<pat::Muon>' )        
    def process(self, event):
        super(SimpleMuonAnalyzer, self).readCollections(event.input)
        event.muons = map(Muon, self.handles['muons'].product())
        
