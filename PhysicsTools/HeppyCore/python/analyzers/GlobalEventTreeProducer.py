from PhysicsTools.HeppyCore.framework.analyzer import Analyzer
from PhysicsTools.HeppyCore.statistics.tree import Tree
from PhysicsTools.HeppyCore.analyzers.ntuple import *

from ROOT import TFile

class GlobalEventTreeProducer(Analyzer):

    def beginLoop(self, setup):
        super(GlobalEventTreeProducer, self).beginLoop(setup)
        self.rootfile = TFile('/'.join([self.dirName,
                                        'tree.root']),
                              'recreate')
        self.tree = Tree( 'events', '')
        bookJet(self.tree, 'sum_all')
        bookJet(self.tree, 'sum_all_gen')
      
    def process(self, event):
        self.tree.reset()
        sum_all = getattr(event, self.cfg_ana.sum_all)
        sum_all_gen = getattr(event, self.cfg_ana.sum_all_gen)
        fillJet(self.tree, 'sum_all', sum_all)
        fillJet(self.tree, 'sum_all_gen', sum_all_gen)
        self.tree.tree.Fill()
        
    def write(self, setup):
        self.rootfile.Write()
        self.rootfile.Close()
        
