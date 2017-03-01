# Copyright (C) 2014 Colin Bernet
# https://github.com/cbernet/heppy/blob/master/LICENSE

import os
import logging

from PhysicsTools.HeppyCore.statistics.counter import Counters
from PhysicsTools.HeppyCore.statistics.average import Averages

class Analyzer(object):
    """Base Analyzer class. Used in Looper.

    Your custom analyzers should inherit from this class
    """

    def __init__(self, cfg_ana, cfg_comp, looperName ):
        """Create an analyzer.

        Parameters (also stored as attributes for later use):
        cfg_ana: configuration parameters for this analyzer (e.g. a pt cut)
        cfg_comp: configuration parameters for the data or MC component (e.g. DYJets)
        looperName: name of the Looper which runs this analyzer.

        Attributes:
        dirName : analyzer directory, where you can write anything you want
        """
        self.class_object = cfg_ana.class_object
        self.instance_label = cfg_ana.instance_label
        self.name = cfg_ana.name
        self.verbose = cfg_ana.verbose
        self.cfg_ana = cfg_ana
        self.cfg_comp = cfg_comp
        self.looperName = looperName
	if hasattr(cfg_ana,"nosubdir") and cfg_ana.nosubdir:
       	    self.dirName = self.looperName
	else:
            self.dirName = '/'.join( [self.looperName, self.name] )
            os.mkdir( self.dirName )


        # this is the main logger corresponding to the looper.
        # each analyzer could also declare its own logger
        self.mainLogger = logging.getLogger( looperName )
        # print self.mainLogger.handlers
        self.beginLoopCalled = False

    def beginLoop(self, setup):
        """Automatically called by Looper, for all analyzers."""
        self.counters = Counters()
        self.averages = Averages()
        self.mainLogger.info( 'beginLoop ' + self.cfg_ana.name )
        self.beginLoopCalled = True

    def endLoop(self, setup):
        """Automatically called by Looper, for all analyzers."""
        #print self.cfg_ana
        self.mainLogger.info( '' )
        self.mainLogger.info( str(self) )
        self.mainLogger.info( '' )

    def process(self, event ):
        """Automatically called by Looper, for all analyzers.
        each analyzer in the sequence will be passed the same event instance.
        each analyzer can access, modify, and store event information, of any type."""
        print self.cfg_ana.name


    def write(self, setup):
        """Called by Looper.write, for all analyzers.
        Just overload it if you have histograms to write."""
        self.counters.write( self.dirName )
        self.averages.write( self.dirName )

    def __str__(self):
        """A multipurpose printout. Should do the job for most analyzers."""
        ana = str( self.cfg_ana )
        count = ''
        ave = ''
        if hasattr(self, 'counters') and len( self.counters.counters ) > 0:
            count = '\n'.join(map(str, self.counters.counters))
        if hasattr(self, 'averages') and len( self.averages ) > 0:
            ave = '\n'.join(map(str, self.averages))
        return '\n'.join( [ana, count, ave] )
