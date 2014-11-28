from PhysicsTools.Heppy.analyzers.core.TreeAnalyzerNumpy import TreeAnalyzerNumpy


class SimpleTreeAnalyzer(TreeAnalyzerNumpy):
    
    def beginLoop(self, setup):
        super(SimpleTreeAnalyzer, self).beginLoop(setup)
        self.bookJet('jet1')
        self.bookJet('jet2')
        self.bookJet('jet1_gen')
        self.bookJet('jet2_gen')
        
    def process(self, event):
        if len(event.jets)>0:
            self.fillJet('jet1', event.jets[0])
            self.fillJet('jet1_gen', event.jets[0].gen)
        if len(event.jets)>1:
            self.fillJet('jet2', event.jets[1])
            self.fillJet('jet2_gen', event.jets[1].gen)
        self.tree.tree.Fill()

    def bookJet(self, name):
        self.tree.var('{name}_pt'.format(name=name) )
    
    def fillJet(self, name, jet):
        if jet:
            self.tree.fill('{name}_pt'.format(name=name), jet.pt())
        
