from PhysicsTools.HeppyCore.framework.analyzer import Analyzer
from PhysicsTools.HeppyCore.statistics.tree import Tree
from PhysicsTools.HeppyCore.analyzers.ntuple import *

from ROOT import TFile

class LeptonTreeProducer(Analyzer):

    def beginLoop(self, setup):
        super(LeptonTreeProducer, self).beginLoop(setup)
        self.rootfile = TFile('/'.join([self.dirName,
                                        'tree.root']),
                              'recreate')
        self.tree = Tree( self.cfg_ana.tree_name,
                          self.cfg_ana.tree_title )
        bookLepton(self.tree, 'lep1')
        bookLepton(self.tree, 'lep2')
        

    def process(self, event):
        self.tree.reset()
        leptons = getattr(event, self.cfg_ana.leptons)
        if len(leptons) > 0:
            fillLepton(self.tree, 'lep1', leptons[0])
        if len(leptons) > 1:
            fillLepton(self.tree, 'lep2', leptons[1])
        self.tree.tree.Fill()
        
        
    def write(self, setup):
        self.rootfile.Write()
        self.rootfile.Close()
        
