from CMGTools.RootTools.analyzers.TreeAnalyzer import TreeAnalyzer
import random


class TestTreeAnalyzer( TreeAnalyzer ):
    '''Just an example. You should create your analyzer on this model.

    One useful technique is to use other analyzers to fill the event with
    what you need. In your TreeAnalyzer, you can simply read the event
    and fill the tree.'''
    def declareVariables(self):
        self.tree.addVar('float', 'gaussianVar')
        self.tree.addVar('float', 'eventWeight')
        self.tree.addVar('int', 'iEv')
        self.tree.book()
        
    def process(self, iEvent, event):
        self.tree.s.gaussianVar = random.gauss( 1, 1 )
        self.tree.s.iEv = event.iEv
        self.tree.s.eventWeight = event.eventWeight
        self.tree.fill()
        
