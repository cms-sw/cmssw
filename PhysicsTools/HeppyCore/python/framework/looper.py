# Copyright (C) 2014 Colin Bernet
# https://github.com/cbernet/heppy/blob/master/LICENSE

import os
import sys
import imp
import logging
import pprint
from math import ceil
from event import Event
import timeit
import resource

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
                  firstEvent=0,
                  nPrint=0,
                  timeReport=False,
                  quiet=False,
                  memCheckFromEvent=-1):
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
        self.logger.propagate = False
        if not quiet: 
            self.logger.addHandler( logging.StreamHandler(sys.stdout) )

        self.cfg_comp = config.components[0]
        self.classes = {}
        self.analyzers = map( self._build, config.sequence )
        self.nEvents = nEvents
        self.firstEvent = firstEvent
        self.nPrint = int(nPrint)
        self.timeReport = [ {'time':0.0,'events':0} for a in self.analyzers ] if timeReport else False
        self.memReportFirstEvent = memCheckFromEvent
        self.memLast=0
        tree_name = None
        if( hasattr(self.cfg_comp, 'tree_name') ):
            tree_name = self.cfg_comp.tree_name
        if len(self.cfg_comp.files)==0:
            errmsg = 'please provide at least an input file in the files attribute of this component\n' + str(self.cfg_comp)
            raise ValueError( errmsg )
        if hasattr(config,"preprocessor") and config.preprocessor is not None :
              self.cfg_comp = config.preprocessor.run(self.cfg_comp,self.outDir,firstEvent,nEvents)
        if hasattr(self.cfg_comp,"options"):
              print self.cfg_comp.files,self.cfg_comp.options
              self.events = config.events_class(self.cfg_comp.files, tree_name,options=self.cfg_comp.options)
        else :
              self.events = config.events_class(self.cfg_comp.files, tree_name)
        if hasattr(self.cfg_comp, 'fineSplit'):
            fineSplitIndex, fineSplitFactor = self.cfg_comp.fineSplit
            if fineSplitFactor > 1:
                if len(self.cfg_comp.files) != 1:
                    raise RuntimeError, "Any component with fineSplit > 1 is supposed to have just a single file, while %s has %s" % (self.cfg_comp.name, self.cfg_comp.files)
                totevents = min(len(self.events),int(nEvents)) if (nEvents and int(nEvents) not in [-1,0]) else len(self.events)
                self.nEvents = int(ceil(totevents/float(fineSplitFactor)))
                self.firstEvent = firstEvent + fineSplitIndex * self.nEvents
                if self.firstEvent + self.nEvents >= totevents:
                    self.nEvents = totevents - self.firstEvent 
                #print "For component %s will process %d events starting from the %d one, ending at %d excluded" % (self.cfg_comp.name, self.nEvents, self.firstEvent, self.nEvents + self.firstEvent)
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
        while True and index < 2000:
            try:
                # print 'mkdir', self.name
                os.mkdir( tmpname )
                break
            except OSError:
                index += 1
                tmpname = '%s_%d' % (name, index)
        if index == 2000:
              raise ValueError( "More than 2000 output folder with same name or 2000 attempts failed, please clean-up, change name or check permissions")
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
        self.logger.info(
            'starting loop at event {firstEvent} '\
                'to process {eventSize} events.'.format(firstEvent=firstEvent,
                                                        eventSize=eventSize))
        self.logger.info( str( self.cfg_comp ) )
        for analyzer in self.analyzers:
            analyzer.beginLoop(self.setup)
        try:
            for iEv in range(firstEvent, firstEvent+eventSize):
                # if iEv == nEvents:
                #     break
                if iEv%100 ==0:
                    # print 'event', iEv
                    if not hasattr(self,'start_time'):
                        print 'event', iEv
                        self.start_time = timeit.default_timer()
                        self.start_time_event = iEv
                    else:
                        print 'event %d (%.1f ev/s)' % (iEv, (iEv-self.start_time_event)/float(timeit.default_timer() - self.start_time))

                self.process( iEv )
                if iEv<self.nPrint:
                    print self.event

        except UserWarning:
            print 'Stopped loop following a UserWarning exception'

        info = self.logger.info
        warning = self.logger.warning
        warning('number of events processed: {nEv}'.format(nEv=iEv+1))
        warning('')
        info( self.cfg_comp )
        info('')        
        for analyzer in self.analyzers:
            analyzer.endLoop(self.setup)
        if self.timeReport:
            allev = max([x['events'] for x in self.timeReport])
            warning("\n      ---- TimeReport (all times in ms; first evt is skipped) ---- ")
            warning("%9s   %9s    %9s   %9s %6s   %s" % ("processed","all evts","time/proc", " time/all", "  [%] ", "analyer"))
            warning("%9s   %9s    %9s   %9s %6s   %s" % ("---------","--------","---------", "---------", " -----", "-------------"))
            sumtime = sum(rep['time'] for rep in self.timeReport)
            passev  = self.timeReport[-1]['events']
            for ana,rep in zip(self.analyzers,self.timeReport):
                timePerProcEv = rep['time']/(rep['events']-1) if rep['events'] > 1 else 0
                timePerAllEv  = rep['time']/(allev-1)         if allev > 1         else 0
                fracAllEv     = rep['time']/sumtime
                warning( "%9d   %9d   %10.2f  %10.2f %5.1f%%   %s" % ( rep['events'], allev, 1000*timePerProcEv, 1000*timePerAllEv, 100.0*fracAllEv, ana.name))
            totPerProcEv = sumtime/(passev-1) if passev > 1 else 0
            totPerAllEv  = sumtime/(allev-1)  if allev > 1  else 0
            warning("%9s   %9s    %9s   %9s   %s" % ("---------","--------","---------", "---------", "-------------"))
            warning("%9d   %9d   %10.2f  %10.2f %5.1f%%   %s" % ( passev, allev, 1000*totPerProcEv, 1000*totPerAllEv, 100.0, "TOTAL"))
            warning("")

    def process(self, iEv ):
        """Run event processing for all analyzers in the sequence.

        This function is called by self.loop,
        but can also be called directly from
        the python interpreter, to jump to a given event.
        """
        self.event = Event(iEv, self.events[iEv], self.setup)
        self.iEvent = iEv
        for i,analyzer in enumerate(self.analyzers):
            if not analyzer.beginLoopCalled:
                analyzer.beginLoop(self.setup)
            start = timeit.default_timer()
            if self.memReportFirstEvent >=0 and iEv >= self.memReportFirstEvent:           
                memNow=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                if memNow > self.memLast :
                   print  "Mem Jump detected before analyzer %s at event %s. RSS(before,after,difference) %s %s %s "%( analyzer.name, iEv, self.memLast, memNow, memNow-self.memLast)
                self.memLast=memNow
            ret = analyzer.process( self.event )
            if self.memReportFirstEvent >=0 and iEv >= self.memReportFirstEvent:           
                memNow=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                if memNow > self.memLast :
                   print "Mem Jump detected in analyzer %s at event %s. RSS(before,after,difference) %s %s %s "%( analyzer.name, iEv, self.memLast, memNow, memNow-self.memLast)
                self.memLast=memNow
            if self.timeReport:
                self.timeReport[i]['events'] += 1
                if self.timeReport[i]['events'] > 0:
                    self.timeReport[i]['time'] += timeit.default_timer() - start
            if ret == False:
                return (False, analyzer.name)
        if iEv<self.nPrint:
            self.logger.info( self.event.__str__() )
        return (True, analyzer.name)

    def write(self):
        """Writes all analyzers.

        See Analyzer.Write for more information.
        """
        for analyzer in self.analyzers:
            analyzer.write(self.setup)
        self.setup.close() 


if __name__ == '__main__':

    import pickle
    import sys
    import os
    if len(sys.argv) == 2 :
        cfgFileName = sys.argv[1]
        pckfile = open( cfgFileName, 'r' )
        config = pickle.load( pckfile )
        comp = config.components[0]
        events_class = config.events_class
    elif len(sys.argv) == 3 :
        cfgFileName = sys.argv[1]
        file = open( cfgFileName, 'r' )
        cfg = imp.load_source( 'cfg', cfgFileName, file)
        compFileName = sys.argv[2]
        pckfile = open( compFileName, 'r' )
        comp = pickle.load( pckfile )
        cfg.config.components=[comp]
        events_class = cfg.config.events_class

    looper = Looper( 'Loop', cfg.config,nPrint = 5)
    looper.loop()
    looper.write()

