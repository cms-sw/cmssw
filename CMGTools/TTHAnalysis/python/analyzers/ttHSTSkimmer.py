from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer

class ttHSTSkimmer( Analyzer ):
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(ttHSTSkimmer,self).__init__(cfg_ana,cfg_comp,looperName)

    def declareHandles(self):
        super(ttHSTSkimmer, self).declareHandles()

    def beginLoop(self, setup):
        super(ttHSTSkimmer,self).beginLoop(setup)
        self.counters.addCounter('events')
        count = self.counters.counter('events')
        count.register('all events')
        count.register('accepted events')


    def process(self, event):
        self.readCollections( event.input )
        self.counters.counter('events').inc('all events')

        if(len(event.selectedLeptons)<1): #ST not defined for events without selected leptons --> skip 
            return False
        if(event.selectedLeptons[0].pt()+event.met.pt()<self.cfg_ana.minST):
            return False
        self.counters.counter('events').inc('accepted events')
        return True
