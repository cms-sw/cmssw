from PhysicsTools.Heppy.analyzers.TreeAnalyzerNumpy import TreeAnalyzerNumpy

def var( tree, varName, type=float ):
    tree.var(varName, type)

def fill( tree, varName, value ):
    tree.fill( varName, value )


class MetTreeProducer( TreeAnalyzerNumpy ):
    def declareVariables(self):
        tr = self.tree
        var( tr, 'u1')
        var( tr, 'u2')
        var( tr, 'met')
        var( tr, 'sumet')
        var( tr, 'zpt')
        var( tr, 'zeta')
        var( tr, 'weight')

    def process(self, iEvent, event):
        
        tr = self.tree
        tr.reset()
        fill( tr, 'u1', event.u1)
        fill( tr, 'u2', event.u2)
        fill( tr, 'met', event.met.pt())
        fill( tr, 'sumet', event.met.sumEt())
        fill( tr, 'zpt', event.diLepton.pt())
        fill( tr, 'zeta', event.diLepton.eta())
        fill( tr, 'weight', event.vertexWeight)
        self.tree.tree.Fill()
       
