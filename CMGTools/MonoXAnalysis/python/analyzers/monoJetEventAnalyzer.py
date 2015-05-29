from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.HeppyCore.utils.deltar import deltaR

        
class monoJetEventAnalyzer( Analyzer ):
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(monoJetEventAnalyzer,self).__init__(cfg_ana,cfg_comp,looperName)

    def declareHandles(self):
        super(monoJetEventAnalyzer, self).declareHandles()

    def beginLoop(self, setup):
        super(monoJetEventAnalyzer,self).beginLoop(setup)
        self.counters.addCounter('events')
        count = self.counters.counter('events')
        count.register('all events')
        count.register('accepted events')

    def process(self, event):
        self.readCollections( event.input )
        self.counters.counter('events').inc('all events')

        if hasattr(self.cfg_ana, 'minJets25'):
            n25 = len([ j for j in event.cleanJets if j.pt() > 25 ])
            if n25 < self.cfg_ana.minJets25: 
                return False

        #self.makeHadTopDecays(event) # here put my calculations of strange variables

        self.counters.counter('events').inc('accepted events')
        return True
