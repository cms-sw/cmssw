# Copyright (C) 2014 Colin Bernet
# https://github.com/cbernet/heppy/blob/master/LICENSE

import ROOT 
ROOT.PyConfig.IgnoreCommandLineOptions = True
import os
import sys
import imp
import logging
import pprint
from math import ceil
from event import Event
import timeit
from PhysicsTools.HeppyCore.framework.exceptions import UserStop
import resource
import json

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
                  memCheckFromEvent=-1,
                  stopFlag = None):
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
    
        stopFlag: it should be a multiprocessing.Value instance, that is set to 1 
                  when this thread, or any other, receives a SIGUSR2 to ask for
                  a graceful job termination. In this case, the looper will also
                  set up a signal handler for SIGUSR2.
                  (if set to None, nothing of all this happens)
        """

        self.config = config
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
        self.stopFlag = stopFlag
        if stopFlag:
            import signal
            def doSigUsr2(sig,frame):
                print 'SIGUSR2 received, signaling graceful stop'
                self.stopFlag.value = 1
            signal.signal(signal.SIGUSR2, doSigUsr2)
        tree_name = None
        if( hasattr(self.cfg_comp, 'tree_name') ):
            tree_name = self.cfg_comp.tree_name
        if len(self.cfg_comp.files)==0:
            errmsg = 'please provide at least an input file in the files attribute of this component\n' + str(self.cfg_comp)
            raise ValueError( errmsg )
        if hasattr(config,"preprocessor") and config.preprocessor is not None :
              self.cfg_comp = config.preprocessor.run(self.cfg_comp,self.outDir,firstEvent,nEvents)
              #in case the preprocessor was run, need to process all events afterwards 
              self.firstEvent = 0
              self.nEvents = None
        if hasattr(self.cfg_comp,"options"):
              print self.cfg_comp.files,self.cfg_comp.options
              self.events = config.events_class(self.cfg_comp.files,
                                                tree_name,
                                                options=self.cfg_comp.options)
        else :
              self.events = config.events_class(self.cfg_comp.files, tree_name)
        if hasattr(self.cfg_comp, 'fineSplit'):
            fineSplitIndex, fineSplitFactor = self.cfg_comp.fineSplit
            if fineSplitFactor > 1:
                if len(self.cfg_comp.files) != 1:
                    raise RuntimeError("Any component with fineSplit > 1 is supposed to have just a single file, while %s has %s" % (self.cfg_comp.name, self.cfg_comp.files))
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
        try: 
            theClass = cfg.class_object
        except AttributeError:
            errfgmt = 'an object of class {cfg_class}'.format(
                cfg_class=cfg.__class__
            )
            if type(cfg) is type:
                errfgmt = 'a class named {class_name}'.format(
                    class_name=cfg.__name__
                )
            err='''
The looper is trying to build an analyzer configured by {errfgmt}. 

Make sure that the configuration object is of class cfg.Analyzer.
            '''.format(errfgmt=errfgmt)
            raise ValueError(err)
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
                # failed to create the directory
                # is it empty?
                if not os.listdir(tmpname):
                    break  # it is, so use it
                else:
                    # if not we append a number to the directory name
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
        self.nEvProcessed = 0
        if nEvents is None or int(nEvents)-firstEvent > len(self.events) :
            nEvents = len(self.events) - firstEvent
        else:
            nEvents = int(nEvents)
        self.logger.info(
            'starting loop at event {firstEvent} '\
                'to process {nEvents} events.'.format(firstEvent=firstEvent,
                                                        nEvents=nEvents))
        self.logger.info( str( self.cfg_comp ) )
        for analyzer in self.analyzers:
            analyzer.beginLoop(self.setup)

        if hasattr(self.events, '__getitem__'):
            # events backend supports indexing, e.g. CMS, FCC, bare root
            for iEv in range(firstEvent, firstEvent+nEvents):
                if iEv%100 == 0:
                    if not hasattr(self,'start_time'):
                        self.logger.info( 'event {iEv}'.format(iEv=iEv))
                        self.start_time = timeit.default_timer()
                        self.start_time_event = iEv
                    else:
                        self.logger.warning( 'event %d (%.1f ev/s)' % (iEv, (iEv-self.start_time_event)/float(timeit.default_timer() - self.start_time)) )
                try:
                    self.process( iEv )
                    self.nEvProcessed += 1
                    if iEv<self.nPrint:
                        self.logger.info(self.event.__str__())
                    if self.stopFlag and self.stopFlag.value:
                        print 'stopping gracefully at event %d' % (iEv)
                        break
                except UserStop as err:
                    print 'Stopped loop following a UserStop exception:'
                    print err
                    break
        else:
            # events backend does not support indexing, e.g. LCIO
            iEv = 0
            for ii, event in enumerate(self.events):
                if ii < firstEvent:
                    continue
                iEv += 1
                if iEv%100 == 0:
                    if not hasattr(self,'start_time'):
                        self.logger.warning( 'event {iEv}'.format(iEv=iEv))
                        self.start_time = timeit.default_timer()
                        self.start_time_event = iEv
                    else:
                        self.logger.info( 'event %d (%.1f ev/s)' % (iEv, (iEv-self.start_time_event)/float(timeit.default_timer() - self.start_time)) )
                try:
                    self.event = Event(iEv, event, self.setup)
                    self.iEvent = iEv
                    self._run_analyzers_on_event()
                    self.nEvProcessed += 1
                    if iEv<self.nPrint:
                        self.logger.info(self.event.__str__())
                    if self.stopFlag and self.stopFlag.value:
                        print 'stopping gracefully at event %d' % (iEv)
                        break
                except UserStop as err:
                    print 'Stopped loop following a UserStop exception:'
                    print err
                    break            
            
        warning = self.logger.warning
        warning('')
        warning( self.cfg_comp )
        warning('')        
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
        logfile = open('/'.join([self.name,'log.txt']),'a')
        logfile.write('number of events processed: {nEv}\n'.format(
            nEv=self.nEvProcessed)
        )
        logfile.close()

    def process(self, iEv ):
        """Run event processing for all analyzers in the sequence.

        This function can be called directly from
        the python interpreter, to jump to a given event and process it.
        """
        if not hasattr(self.events, '__getitem__'):
            msg = '''
Your events backend, of type 
{evclass}
does not support indexing. 
Therefore, you cannot directly access a given event using Loop.process.
However, you may still iterate on your events using Loop.loop, 
possibly skipping a number of events at the beginning.
'''.format(evclass=self.events.__class__)
            raise TypeError(msg)
        self.event = Event(iEv, self.events[iEv], self.setup)            
        self.iEvent = iEv
        return self._run_analyzers_on_event()

    def _run_analyzers_on_event(self):
        '''Run all analysers on the current event, self.event. 
        Returns a tuple (success?, last_analyzer_name).
        '''
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
    from PhysicsTools.HeppyCore.framework.heppy_loop import _heppyGlobalOptions
    from optparse import OptionParser
    parser = OptionParser(usage='%prog cfgFileName compFileName [--options=optFile.json]')
    parser.add_option('--options',dest='options',default='',help='options json file')
    (options,args) = parser.parse_args()

    if options.options!='':
        jsonfilename = options.options
        jfile = open (jsonfilename, 'r')
        opts=json.loads(jfile.readline())
        for k,v in opts.iteritems():
            _heppyGlobalOptions[k]=v
        jfile.close()

    if len(args) == 1 :
        cfgFileName = args[0]
        pckfile = open( cfgFileName, 'r' )
        config = pickle.load( pckfile )
        comp = config.components[0]
        events_class = config.events_class
    elif len(args) == 2 :
        cfgFileName = args[0]
        file = open( cfgFileName, 'r' )
        cfg = imp.load_source( 'cfg', cfgFileName, file)
        compFileName = args[1]
        pckfile = open( compFileName, 'r' )
        comp = pickle.load( pckfile )
        cfg.config.components=[comp]
        events_class = cfg.config.events_class

    looper = Looper( 'Loop', cfg.config,nPrint = 5)
    looper.loop()
    looper.write()

