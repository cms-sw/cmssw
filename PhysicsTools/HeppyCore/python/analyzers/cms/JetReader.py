from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from PhysicsTools.HeppyCore.particles.cms.jet import Jet

import math

class JetReader(Analyzer):
    
    def declareHandles(self):
        super(JetReader, self).declareHandles()
        self.handles['jets'] = AutoHandle(
            self.cfg_ana.jets, 
            'std::vector<reco::PFJet>'
            )
        self.handles['gen_jets'] = AutoHandle(
            self.cfg_ana.gen_jets, 
            'std::vector<reco::GenJet>'
            )

    def process(self, event):
        self.readCollections(event.input)
        store = event.input
        genj = self.handles['gen_jets'].product()
        genj = [jet for jet in genj if jet.pt()>self.cfg_ana.gen_jet_pt]
        gen_jets = map(Jet, genj)
        event.gen_jets = sorted( gen_jets,
                                 key = lambda ptc: ptc.pt(), reverse=True )  
        
        for jet in event.gen_jets:
            jet.constituents.validate(jet.e())

        pfj = self.handles['jets'].product()
        pfj = [jet for jet in pfj if jet.pt()>self.cfg_ana.jet_pt]
        jets = map(Jet, pfj)
        event.cms_jets = sorted( jets,
                                 key = lambda ptc: ptc.pt(), reverse=True )  
        
        # for jet in event.cms_jets:
        #     jet.constituents.validate(jet.e())
