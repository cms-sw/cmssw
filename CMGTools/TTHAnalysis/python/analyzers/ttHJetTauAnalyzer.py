from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
#from CMGTools.RootTools.fwlite.AutoHandle import AutoHandle


class ttHJetTauAnalyzer( Analyzer ):
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(ttHJetTauAnalyzer,self).__init__(cfg_ana,cfg_comp,looperName)

    def declareHandles(self):
        super(ttHJetTauAnalyzer, self).declareHandles()
      
    def beginLoop(self, setup):
        super(ttHJetTauAnalyzer,self).beginLoop(setup)

    def findNonTauJets(self, jets):
        iJetNoTau =  []
        for ij in xrange(len(jets)):
            if not jets[ij].taus:
                iJetNoTau.append(ij)
                
        return iJetNoTau

    def process(self, event):
        self.readCollections(event.input)
        
        event.jetsNonTauIdx = self.findNonTauJets(event.cleanJets)

        return True
