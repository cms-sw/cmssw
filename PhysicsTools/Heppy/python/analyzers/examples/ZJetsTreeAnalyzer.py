from PhysicsTools.Heppy.analyzers.core.TreeAnalyzerNumpy import TreeAnalyzerNumpy
import ntuple

class ZJetsTreeAnalyzer(TreeAnalyzerNumpy):
    
    def beginLoop(self, setup):
        super(ZJetsTreeAnalyzer, self).beginLoop(setup)
        ntuple.bookParticle(self.tree, 'jet1')
        ntuple.bookParticle(self.tree, 'jet1_gen')
        ntuple.bookParticle(self.tree, 'jet2')
        ntuple.bookParticle(self.tree, 'jet2_gen')
        ntuple.bookParticle(self.tree, 'dimuon')
        ntuple.bookParticle(self.tree, 'dimuon_leg1')
        ntuple.bookParticle(self.tree, 'dimuon_leg2')

        
    def process(self, event):
        self.tree.reset()
        if len(event.jets)>0:
            ntuple.fillParticle(self.tree, 'jet1', event.jets[0])
            if event.jets[0].gen:
                ntuple.fillParticle(self.tree, 'jet1_gen', event.jets[0].gen)
        if len(event.jets)>1:
            ntuple.fillParticle(self.tree, 'jet2', event.jets[1])
            if event.jets[1].gen:
                ntuple.fillParticle(self.tree, 'jet2_gen', event.jets[1].gen)
        if len(event.dimuons)>1:
            ntuple.fillParticle(self.tree, 'dimuon', event.dimuons[0])
            ntuple.fillParticle(self.tree, 'dimuon_leg1', event.dimuons[0].leg1)
            ntuple.fillParticle(self.tree, 'dimuon_leg2', event.dimuons[0].leg2)
        self.tree.tree.Fill()

        
