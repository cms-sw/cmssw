from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.HeppyCore.framework.event import Event
from PhysicsTools.HeppyCore.statistics.counter import Counter, Counters
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle

class ttHmllSkimmer( Analyzer ):
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(ttHmllSkimmer,self).__init__(cfg_ana,cfg_comp,looperName)

    def declareHandles(self):
        super(ttHmllSkimmer, self).declareHandles()

    def beginLoop(self,setup):
        super(ttHmllSkimmer,self).beginLoop(setup)
        self.counters.addCounter('events')
        count = self.counters.counter('events')
        count.register('all events')
        count.register('pass gen Zll skim')
        count.register('pass reco Zll skim')

    def makeZs(self, event, maxLeps, lepId):
        event.bestZ = [ 0., -1,-1 ]
        nlep = len(event.selectedLeptons)
        for i,l1 in enumerate(event.selectedLeptons):
            for j in range(i+1,nlep):
                if j >= maxLeps: break
                l2 = event.selectedLeptons[j]
                if l1.pdgId() == -l2.pdgId() and abs(l1.pdgId()) in lepId:
                    zmass = (l1.p4() + l2.p4()).M()
                    if event.bestZ[0] == 0 or abs(zmass - 91.188) < abs(event.bestZ[0] - 91.188):
                        event.bestZ = [ zmass, i, j ]

    def process(self, event):
        self.readCollections( event.input )
        self.counters.counter('events').inc('all events')

        if self.cfg_ana.doZGen and len(event.genleps)==2:
            if event.genleps[0].pdgId() == - event.genleps[1].pdgId() and abs(event.genleps[0].pdgId()) in self.cfg_ana.lepId :
                if (event.genleps[0].sourceId==23 and event.genleps[1].sourceId==23) :
                    self.counters.counter('events').inc('pass gen Zll skim')
                    return True

        if self.cfg_ana.doZReco:
            self.makeZs( event, self.cfg_ana.maxLeps, self.cfg_ana.lepId)
            if event.bestZ[0] > self.cfg_ana.massMin and event.bestZ[0] < self.cfg_ana.massMax:
                event.zll_p4 = event.selectedLeptons[event.bestZ[1]].p4() + event.selectedLeptons[event.bestZ[2]].p4()
                self.counters.counter('events').inc('pass reco Zll skim')
                return True

        #If none of the Z selection return veto
        return False
