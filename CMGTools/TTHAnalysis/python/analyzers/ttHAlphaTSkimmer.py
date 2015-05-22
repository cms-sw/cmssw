
from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.HeppyCore.framework.event import Event
from PhysicsTools.HeppyCore.statistics.counter import Counter, Counters
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle

class ttHAlphaTSkimmer( Analyzer ):
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(ttHAlphaTSkimmer,self).__init__(cfg_ana,cfg_comp,looperName)

    def declareHandles(self):
        super(ttHAlphaTSkimmer, self).declareHandles()

    def beginLoop(self,setup):
        super(ttHAlphaTSkimmer,self).beginLoop(setup)
        self.counters.addCounter('events')
        count = self.counters.counter('events')
        count.register('all events')
        count.register('pass forwardJetVeto')
        count.register('pass MHT/MET cut')
        count.register('pass alphaTCuts')
        count.register('accepted events')


    def process(self, event):
        self.readCollections( event.input )
        self.counters.counter('events').inc('all events')

        #Veto forward jets that have passed the jet requirement
        if self.cfg_ana.forwardJetVeto and len(event.cleanJetsFwd) > 0:
            return False
        self.counters.counter('events').inc('pass forwardJetVeto')

        #Veto events that don't pass the MHT/MET cut
        if getattr(event, self.cfg_ana.mhtDivMetCut[0])/getattr(event, self.cfg_ana.mhtDivMetCut[1]).pt() > self.cfg_ana.mhtDivMetCut[2]:
            return False
        self.counters.counter('events').inc('pass MHT/MET cut')
            
        #Check if the event passes the alphaT cut

        if self.cfg_ana.invertAlphaT: #This is for the multijet enriched control region

            for aTCut in self.cfg_ana.alphaTCuts:
                if event.alphaT < aTCut[0] and getattr(event,self.cfg_ana.htJet) >= aTCut[1] and getattr(event,self.cfg_ana.htJet) < aTCut[2]:
                    self.counters.counter('events').inc('pass alphaTCuts')
                    self.counters.counter('events').inc('accepted events')
                    return True

        else:

            for aTCut in self.cfg_ana.alphaTCuts:
                if event.alphaT > aTCut[0] and getattr(event,self.cfg_ana.htJet) >= aTCut[1] and getattr(event,self.cfg_ana.htJet) < aTCut[2]:
                    self.counters.counter('events').inc('pass alphaTCuts')
                    self.counters.counter('events').inc('accepted events')
                    return True

        #If none of the alphaT cuts are passed, veto the event
        return False
