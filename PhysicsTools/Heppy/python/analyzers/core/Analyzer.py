import os 
import logging 

from PhysicsTools.HeppyCore.framework.analyzer import Analyzer as CoreAnalyzer

class Analyzer(CoreAnalyzer):
    '''Base Analyzer class. Used in Looper.'''
    
    def declareHandles(self):
        self.handles = {}
        self.mchandles = {}

    def beginLoop(self):
        '''Automatically called by Looper, for all analyzers.'''
        super(Analyzer, self).beginLoop()
        self.declareHandles()
        

    def process(self, iEvent, event ):
        '''Automatically called by Looper, for all analyzers.
        each analyzer in the sequence will be passed the same event instance.
        each analyzer can access, modify, and store event information, of any type.'''
        print self.cfg_ana.name
        self.readCollections( iEvent )

    def readCollections(self, iEvent ):
        '''You must call this function at the beginning of the process
        function of your child analyzer.'''
        # if not self.beginLoopCalled:
        #    # necessary in case the user calls process to go straight to a given event, before looping
        #    self.beginLoop()
        for str,handle in self.handles.iteritems():
            handle.Load( iEvent )
        if self.cfg_comp.isMC:
            for str,handle in self.mchandles.iteritems():
                handle.Load( iEvent )
