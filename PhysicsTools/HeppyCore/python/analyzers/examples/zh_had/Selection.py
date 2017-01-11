from PhysicsTools.HeppyCore.framework.analyzer import Analyzer
from PhysicsTools.HeppyCore.statistics.counter import Counter

class Selection(Analyzer):

    def beginLoop(self, setup):
        super(Selection, self).beginLoop(setup)
        self.counters.addCounter('cut_flow') 
        self.counters['cut_flow'].register('All events')
        self.counters['cut_flow'].register('No lepton')
        self.counters['cut_flow'].register('4 jets')
        self.counters['cut_flow'].register('4 jets with E>15')
        self.counters['cut_flow'].register('2 b jets')
    
    def process(self, event):
        self.counters['cut_flow'].inc('All events')
        if len(event.sel_iso_leptons) > 0:
            return True # could return False to stop processing
        self.counters['cut_flow'].inc('No lepton')
        jets = getattr(event, self.cfg_ana.input_jets)        
        if len(jets) < 4:
            return True
        self.counters['cut_flow'].inc('4 jets')
        if min(jet.e() for jet in jets) >= 15.:
            self.counters['cut_flow'].inc('4 jets with E>15')
        bjets = [jet for jet in jets if jet.tags['b']]
        if len(bjets) >= 2:
            self.counters['cut_flow'].inc('2 b jets')
