# Copyright (C) 2014 Colin Bernet
# https://github.com/cbernet/heppy/blob/master/LICENSE

import os
import sys
import imp
# import copy
import logging
import pprint
from platform import platform 
from event import Event


class Setup(object):
    '''The Looper creates a Setup object to hold information relevant during 
    the whole process, such as the process configuration obtained from 
    the configuration file, or services that can be used by several analyzers.

    The user may freely attach new information to the setup object, 
    as long as this information is relevant during the whole process. 
    If the information is event specific, it should be attached to the event 
    object instead.
    ''' 
    def __init__(self, config, services):
        '''
        Create a Setup object. 
        
        parameters: 
        
        config: configuration object from the configuration file
        
        services: dictionary of services indexed by service name.
        The service name has the form classObject_instanceLabel 
        as in this example: 
        <base_heppy_path>.framework.services.tfile.TFileService_myhists
        To find out about the service name of a given service, 
        load your configuration file in python, and print the service. 
        '''
        self.config = config
        self.services = services
        
    def close(self):
        '''Stop all services'''
        for service in self.services.values():
            service.stop()
        

class Looper(object):
    """Creates a set of analyzers, and schedules the event processing."""

    def __init__( self, name,
                  config, 
                  nEvents=None,
                  firstEvent=0, nPrint=0 ):
        """Handles the processing of an event sample.
        An Analyzer is built for each Config.Analyzer present
        in sequence. The Looper can then be used to process an event,
        or a collection of events.

        Parameters:
        name    : name of the Looper, will be used as the output directory name
        config  : process configuration information, see Config
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

        self.cfg_comp = config.components[0]
        self.classes = {}
        self.analyzers = map( self._build, config.sequence )
        self.nEvents = nEvents
        self.firstEvent = firstEvent
        self.nPrint = int(nPrint)
        tree_name = None
        if( hasattr(self.cfg_comp, 'tree_name') ):
            tree_name = self.cfg_comp.tree_name
        if len(self.cfg_comp.files)==0:
            errmsg = 'please provide at least an input file in the files attribute of this component\n' + str(self.cfg_comp)
            raise ValueError( errmsg )
        self.events = config.events_class(self.cfg_comp.files, tree_name)
        # self.event is set in self.process
        self.event = None
        services = dict()
        for cfg_serv in config.services:
            service = self._build(cfg_serv)
            services[cfg_serv.name] = service
        # would like to provide a copy of the config to the setup,
        # so that analyzers cannot modify the config of other analyzers. 
        # but cannot copy the autofill config.
        self.setup = Setup(config, services)

    def _build(self, cfg):
        theClass = cfg.class_object
        obj = theClass( cfg, self.cfg_comp, self.outDir )
        return obj
        
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


    def loop(self):
        """Loop on a given number of events.

        At the beginning of the loop, 
        Analyzer.beginLoop is called for each Analyzer.
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
        self.logger.warning(
            'starting loop at event {firstEvent} '\
                'to process {eventSize} events.'.format(firstEvent=firstEvent,
                                                        eventSize=eventSize))
        self.logger.warning( str( self.cfg_comp ) )
        for analyzer in self.analyzers:
            analyzer.beginLoop(self.setup)
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
            analyzer.endLoop(self.setup)
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
        self.event = Event(iEv, self.events[iEv], self.setup)
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
        self.setup.close() 
        pass


if __name__ == '__main__':

    import pickle
    import sys
    import os

    cfgFileName = sys.argv[1]
    pckfile = open( cfgFileName, 'r' )
    config = pickle.load( pckfile )
    comp = config.components[0]
    events_class = config.events_class
    looper = Looper( 'Loop', comp,
                     config.sequence,
                     config.services,
                     events_class, 
                     nPrint = 5)
    looper.loop()
    looper.write()
