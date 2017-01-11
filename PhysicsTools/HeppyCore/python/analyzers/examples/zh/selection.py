from PhysicsTools.HeppyCore.framework.analyzer import Analyzer
from PhysicsTools.HeppyCore.statistics.counter import Counter

class Selection(Analyzer):

    def beginLoop(self, setup):
        super(Selection, self).beginLoop(setup)
        self.counters.addCounter('cut_flow') 
        self.counters['cut_flow'].register('All events')
        self.counters['cut_flow'].register('At least 2 leptons')
        self.counters['cut_flow'].register('Both leptons e>30')
    
    def process(self, event):
        self.counters['cut_flow'].inc('All events')
        if len(event.sel_iso_leptons)<2:
            return True # could return False to stop processing
        self.counters['cut_flow'].inc('At least 2 leptons')
        if event.sel_iso_leptons[1].e()>30.:
            self.counters['cut_flow'].inc('Both leptons e>30')
        return True
