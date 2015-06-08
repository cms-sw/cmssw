from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer

class monoJetSkimmer( Analyzer ):
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(monoJetSkimmer,self).__init__(cfg_ana,cfg_comp,looperName)
        self.metCut = cfg_ana.metCut if hasattr(cfg_ana, 'metCut') else []

    def declareHandles(self):
        super(monoJetSkimmer, self).declareHandles()

    def beginLoop(self, setup):
        super(monoJetSkimmer,self).beginLoop(setup)
        self.counters.addCounter('events')
        count = self.counters.counter('events')
        count.register('all events')
        count.register('pass jetPtCuts')
        count.register('pass jetVeto')
        count.register('pass met')
        count.register('accepted events')

    def process(self, event):
        self.readCollections( event.input )
        self.counters.counter('events').inc('all events')

        jets = getattr(event, self.cfg_ana.jets)
        for i,ptCut in enumerate(self.cfg_ana.jetPtCuts):
            if len(jets) <= i or jets[i].pt() <= ptCut:
                return False
        self.counters.counter('events').inc('pass jetPtCuts')
        
        if float(self.cfg_ana.metCut) > 0 and event.metNoMu.pt() <= self.cfg_ana.metCut:
            return False
        self.counters.counter('events').inc('pass met')

        self.counters.counter('events').inc('accepted events')
        return True
