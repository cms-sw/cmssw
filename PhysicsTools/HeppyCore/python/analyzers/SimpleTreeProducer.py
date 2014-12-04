from PhysicsTools.HeppyCore.framework.analyzer import Analyzer
from PhysicsTools.HeppyCore.statistics.tree import Tree
from ROOT import TFile

class SimpleTreeProducer(Analyzer):

    def beginLoop(self, setup):
        super(SimpleTreeProducer, self).beginLoop(setup)
        self.rootfile = TFile('/'.join([self.dirName,
                                        'simple_tree.root']),
                              'recreate')
        self.tree = Tree( self.cfg_ana.tree_name,
                          self.cfg_ana.tree_title )
        self.tree.var('test_variable')

    def process(self, event):
        self.tree.fill('test_variable', event.input.var1)
        self.tree.tree.Fill()

    def write(self, setup):
        self.rootfile.Write()
        self.rootfile.Close()
        
