# Copyright (C) 2014 Colin Bernet
# https://github.com/cbernet/heppy/blob/master/LICENSE

import os
import sys
import imp
import logging
import pprint
from chain import Chain as Events
if os.environ.get('HEPPY_FCC', None) is not None:
    from eventsalbers import Events
elif os.environ.get('HEPPY_CMS', None) is not None:
    assert(False)
from event import Event

class Looper(object):
    """Creates a set of analyzers, and schedules the event processing."""

    def __init__( self, name, cfg_comp, sequence, nEvents=None,
                  firstEvent=0, nPrint=0):
        """Handles the processing of an event sample.
        An Analyzer is built for each Config.Analyzer present
        in sequence. The Looper can then be used to process an event,
        or a collection of events.

        Parameters:
        name    : name of the Looper, will be used as the output directory name
        cfg_comp: information for the input sample, see Config
        sequence: an ordered list of Config.Analyzer
        nEvents : number of events to process. Defaults to all.
        firstEvent : first event to process. Defaults to the first one.
        nPrint  : number of events to print at the beginning
        """

        self.name = self._prepareOutput(name)
        self.outDir = self.name
        self.logger = logging.getLogger( self.name )
        self.logger.addHandler(logging.FileHandler('/'.join([self.name,
                                                             'log.txt'])))
        self.logger.addHandler( logging.StreamHandler(sys.stdout) )

        self.cfg_comp = cfg_comp
        self.classes = {}
        self.analyzers = map( self._buildAnalyzer, sequence )
        self.nEvents = nEvents
        self.firstEvent = firstEvent
        self.nPrint = int(nPrint)
        tree_name = None
        if( hasattr(self.cfg_comp, 'tree_name') ):
            tree_name = self.cfg_comp.tree_name
        self.events = Events(self.cfg_comp.files, tree_name)
        # self.event is set in self.process
        self.event = None

    def _prepareOutput(self, name):
        index = 0
        tmpname = name
        while True:
            try:
                # print 'mkdir', self.name
                os.mkdir( tmpname )
                break
            except OSError:
                index += 1
                tmpname = '%s_%d' % (name, index)
        return tmpname

    def _buildAnalyzer(self, cfg_ana):
        obj = None
        className = cfg_ana.name.split('_')[0]
        theClass = None
        try:
            # obviously, can't load a module twice
            # so keep track of the needed classes,
            # instead several instances are built
            theClass = self.classes[className]
            print 'found class', theClass
            obj = theClass( cfg_ana, self.cfg_comp, self.outDir )
        except KeyError:
            file = None
            try:
                file, path, desc = imp.find_module( className )
                mod  = imp.load_module( className ,
                                        file, path, desc )
                # getting the analyzer class object
                theClass = mod.__dict__[ className ]
                self.classes[className] = theClass
                # creating an analyzer
                #if hasattr( cfg_ana, 'instanceName'):
                #    cfg_ana.name = cfg_ana.instanceName
                print 'loading class', theClass
                print '  from', file
                obj = theClass( cfg_ana, self.cfg_comp, self.outDir )
            finally:
                try:
                    file.close()
                except AttributeError:
                    print 'problem loading module', cfg_ana.name
                    print 'please make sure that the module name is correct.'
                    print 'if it is, is this module in your path, as defined below?'
                    pprint.pprint( sorted( sys.path ))
        return obj

    def loop(self):
        """Loop on a given number of events.

        At the beginning of the loop, Analyzer.beginLoop is called for each Analyzer.
        At each event, self.process is called.
        At the end of the loop, Analyzer.endLoop is called.
        """
        nEvents = self.nEvents
        firstEvent = self.firstEvent
        iEv = firstEvent
        if nEvents is None or int(nEvents) > len(self.events) :
            nEvents = len(self.events)
        else:
            nEvents = int(nEvents)
        eventSize = nEvents
        self.logger.warning('starting loop at event {firstEvent} '\
                            'to process {eventSize} events.'.format(firstEvent=firstEvent,
                                                                    eventSize=eventSize))
        self.logger.warning( str( self.cfg_comp ) )
        for analyzer in self.analyzers:
            analyzer.beginLoop()

        try:
            for iEv in range(firstEvent, firstEvent+eventSize):
                # if iEv == nEvents:
                #     break
                if iEv%100 ==0:
                    print 'event', iEv
                self.process( iEv )
                if iEv<self.nPrint:
                    print self.event
        except UserWarning:
            print 'Stopped loop following a UserWarning exception'
        for analyzer in self.analyzers:
            analyzer.endLoop()
        warn = self.logger.warning
        warn('')
        warn( self.cfg_comp )
        warn('')
        warn('number of events processed: {nEv}'.format(nEv=iEv+1))

    def process(self, iEv ):
        """Run event processing for all analyzers in the sequence.

        This function is called by self.loop,
        but can also be called directly from
        the python interpreter, to jump to a given event.
        """
        self.event = Event(iEv, self.events[iEv])
        self.iEvent = iEv
        for analyzer in self.analyzers:
            if not analyzer.beginLoopCalled:
                analyzer.beginLoop()
            if analyzer.process( self.event ) == False:
                return (False, analyzer.name)
        return (True, analyzer.name)

    def write(self):
        """Writes all analyzers.

        See Analyzer.Write for more information.
        """
        for analyzer in self.analyzers:
            analyzer.write()
        pass


if __name__ == '__main__':

    import pickle
    import sys
    import os

    cfgFileName = sys.argv[1]
    pckfile = open( cfgFileName, 'r' )
    config = pickle.load( pckfile )
    comp = config.components[0]
    looper = Looper( 'Loop', comp,
                     config.sequence, nPrint = 5)
    looper.loop()
    looper.write()
