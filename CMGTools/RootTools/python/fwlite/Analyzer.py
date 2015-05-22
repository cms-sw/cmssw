import os 
import logging 


from CMGTools.RootTools.statistics.Counter import Counters
from CMGTools.RootTools.statistics.Average import Averages

class Analyzer(object):
    '''Base Analyzer class. Used in Looper.'''
    
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        '''Create an analyzer.

        cfg_ana: configuration parameters for this analyzer (e.g. a pt cut)
        cfg_comp: configuration parameters for the data or MC component (e.g. DYJets)
        looperName: name of the Looper which runs this analyzer.

        See Looper and Config for more information.

        You should inherit from this class.
        Interesting examples of child analyzers are:
          DiLeptonAnalyzer, TauMuAnalyzer, VertexAnalyzer, H2TauTauEventSorter. 
        '''
        self.name = cfg_ana.name
        self.verbose = cfg_ana.verbose
        self.cfg_ana = cfg_ana
        self.cfg_comp = cfg_comp
        self.looperName = looperName
        self.dirName = '/'.join( [self.looperName, self.name] )
        os.mkdir( self.dirName )

        # this is the main logger corresponding to the looper.
        # each analyzer could also declare its own logger
        self.mainLogger = logging.getLogger( looperName )
        self.beginLoopCalled = False

    def declareHandles(self):
        self.handles = {}
        self.mchandles = {}
        self.embhandles = {}

    def beginLoop(self):
        '''Automatically called by Looper, for all analyzers.'''
        self.declareHandles()
        self.counters = Counters()
        self.averages = Averages()
        self.mainLogger.warning( 'beginLoop ' + self.cfg_ana.name ) 
        self.beginLoopCalled = True
        
    def endLoop(self):
        '''Automatically called by Looper, for all analyzers.'''
        #print self.cfg_ana
        self.mainLogger.warning( '' )
        self.mainLogger.warning( str(self) )
        self.mainLogger.warning( '' )

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
        if self.cfg_comp.isEmbed:
            for str,handle in self.embhandles.iteritems():
                handle.Load( iEvent )

    def write(self):
        '''Called by Looper.write, for all analyzers.
        Just overload it if you have histograms to write.'''
        # print 'writing not implemented for', self.cfg_ana.name 
        self.counters.write( self.dirName )
        self.averages.write( self.dirName )

    def __str__(self):
        '''A multipurpose printout. Should do the job for most analyzers.'''
        ana = str( self.cfg_ana )
        count = ''
        ave = ''
        if hasattr(self, 'counters') and len( self.counters.counters ) > 0:
            count = '\n'.join(map(str, self.counters.counters))
        if hasattr(self, 'averages') and len( self.averages ) > 0:
            ave = '\n'.join(map(str, self.averages))
        return '\n'.join( [ana, count, ave] )
        

