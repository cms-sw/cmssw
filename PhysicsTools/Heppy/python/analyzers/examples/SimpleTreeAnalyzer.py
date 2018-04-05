from PhysicsTools.Heppy.analyzers.core.TreeAnalyzerNumpy import TreeAnalyzerNumpy
import ntuple

class ZJetsTreeAnalyzer(TreeAnalyzerNumpy):
    
    def beginLoop(self, setup):
        super(ZJetsTreeAnalyzer, self).beginLoop(setup)
        ntuple.bookJet('jet1')
        ntuple.bookJet('jet2')
        ntuple.bookZ('dimuon')

        
    def process(self, event):
        if len(event.jets)>0:
            ntuple.fillJet('jet1', event.jets[0])
        if len(event.jets)>1:
            ntuple.fillJet('jet2', event.jets[1])
        if len(event.dimuons>1):
            ntuple.fillZ('dimuon', event.dimuons[0])
        ntuple.tree.tree.Fill()

        
