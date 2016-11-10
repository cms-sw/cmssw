from PhysicsTools.HeppyCore.framework.analyzer import Analyzer
from PhysicsTools.HeppyCore.statistics.tree import Tree
from PhysicsTools.HeppyCore.analyzers.ntuple import *

from ROOT import TFile

class IsoParticleTreeProducer(Analyzer):

    def beginLoop(self, setup):
        super(IsoParticleTreeProducer, self).beginLoop(setup)
        self.rootfile = TFile('/'.join([self.dirName,
                                        'tree.root']),
                              'recreate')
        self.tree = Tree( self.cfg_ana.tree_name,
                          self.cfg_ana.tree_title )
        bookIsoParticle(self.tree, 'ptc')

    def process(self, event):
        self.tree.reset()
        leptons = getattr(event, self.cfg_ana.leptons)
        pdgids = [211, 22, 130]
        for lepton in leptons:
            for pdgid in pdgids:
                iso = getattr(lepton, 'iso_{pdgid:d}'.format(pdgid=pdgid))
                for ptc in iso.on_ptcs:
                    self.tree.reset()
                    fillIsoParticle(self.tree, 'ptc', ptc, lepton)
                    self.tree.tree.Fill()
        
    def write(self, setup):
        self.rootfile.Write()
        self.rootfile.Close()
        
