
from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.HeppyCore.framework.event import Event
from PhysicsTools.HeppyCore.statistics.counter import Counter, Counters
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle

class ttHAlphaTControlSkimmer( Analyzer ):
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(ttHAlphaTControlSkimmer,self).__init__(cfg_ana,cfg_comp,looperName)

    def declareHandles(self):
        super(ttHAlphaTControlSkimmer, self).declareHandles()

    def beginLoop(self,setup):
        super(ttHAlphaTControlSkimmer,self).beginLoop(setup)
        self.counters.addCounter('events')
        count = self.counters.counter('events')
        count.register('all events')
        count.register('pass mtw cuts')
        count.register('pass mll cuts')
        count.register('pass lepton deltaR cuts')
        count.register('pass photon deltaR cuts')
        count.register('accepted events')


    def process(self, event):
        self.readCollections( event.input )
        self.counters.counter('events').inc('all events')

        #Perform W mass MTW cut
        if event.mtw < self.cfg_ana.mtwCut[0] or event.mtw > self.cfg_ana.mtwCut[1]:
            return False
        self.counters.counter('events').inc('pass mtw cuts')

        #Perform Z mass MLL cut
        if event.mll < self.cfg_ana.mllCut[0] or event.mll > self.cfg_ana.mllCut[1]:
            return False
        self.counters.counter('events').inc('pass mll cuts')
            
        #Perform deltaR cuts 
        for dR in event.minDeltaRLepJet:
            if dR < self.cfg_ana.lepDeltaRCut:
                return False
        self.counters.counter('events').inc('pass lepton deltaR cuts')

        for dR in event.minDeltaRPhoJet:
            if dR < self.cfg_ana.photonDeltaRCut:
                return False
        self.counters.counter('events').inc('pass photon deltaR cuts')

        #If everything passed return true
        self.counters.counter('events').inc('accepted events')
        return True

