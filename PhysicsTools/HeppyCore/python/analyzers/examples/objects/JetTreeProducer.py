'''some module doc'''

from PhysicsTools.HeppyCore.framework.analyzer import Analyzer
from PhysicsTools.HeppyCore.statistics.tree import Tree
from PhysicsTools.HeppyCore.analyzers.ntuple import *

from ROOT import TFile

class JetTreeProducer(Analyzer):
    '''Some class doc'''

    def beginLoop(self, setup):
        super(JetTreeProducer, self).beginLoop(setup)
        self.rootfile = TFile('/'.join([self.dirName,
                                        'jet_tree.root']),
                              'recreate')
        self.tree = Tree( self.cfg_ana.tree_name,
                          self.cfg_ana.tree_title )
        bookJet(self.tree, 'jet1')
        bookJet(self.tree, 'jet1_gen')
        bookJet(self.tree, 'jet2')
        bookJet(self.tree, 'jet2_gen')
        var(self.tree, 'event')
        var(self.tree, 'lumi')
        var(self.tree, 'run')
 

    def process(self, event):
        self.tree.reset()
        if hasattr(event, 'eventId'): 
            fill(self.tree, 'event', event.eventId)
            fill(self.tree, 'lumi', event.lumi)
            fill(self.tree, 'run', event.run)
        jets = getattr(event, self.cfg_ana.jets)
        if( len(jets)>0 ):
            jet = jets[0]
            comp211 = jet.constituents.get(211, None)
            if comp211: 
                if comp211.num==2:
                    import pdb; pdb.set_trace()
            fillJet(self.tree, 'jet1', jet)
            if jet.match:
                fillJet(self.tree, 'jet1_gen', jet.match)
                # if jet.e()/jet.match.e() > 2.:
                #     import pdb; pdb.set_trace()
        if( len(jets)>1 ):
            jet = jets[1]
            fillJet(self.tree, 'jet2', jet)
            if jet.match:
                fillJet(self.tree, 'jet2_gen', jet.match)
        self.tree.tree.Fill()
        
        
    def write(self, setup):
        self.rootfile.Write()
        self.rootfile.Close()
        
