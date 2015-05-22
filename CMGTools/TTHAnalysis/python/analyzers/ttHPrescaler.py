from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer

class ttHPrescaler( Analyzer ):
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(ttHPrescaler,self).__init__(cfg_ana,cfg_comp,looperName)
        self.prescaleFactor = cfg_ana.prescaleFactor
        self.events = 0

    def declareHandles(self):
        super(ttHPrescaler, self).declareHandles()

    def beginLoop(self, setup):
        super(ttHPrescaler,self).beginLoop(setup)
        self.counters.addCounter('events')
        count = self.counters.counter('events')
        count.register('all events')
        count.register('accepted events')


    def process(self, event):
        self.events += 1
        self.counters.counter('events').inc('all events')
        if (self.events % self.prescaleFactor == 1):
            self.counters.counter('events').inc('accepted events')
            return True
        else:
            return False

