#!/usr/bin/env python
# Copyright (C) 2014 Colin Bernet
# https://github.com/cbernet/heppy/blob/master/LICENSE

import os
import shutil
import glob
import sys
import imp
import copy
import multiprocessing 
from pprint import pprint

# import root in batch mode if "-i" is not among the options
if "-i" not in sys.argv:
    oldv = sys.argv[:]
    sys.argv = [ "-b-"]
    import ROOT
    ROOT.gROOT.SetBatch(True)
    sys.argv = oldv


from PhysicsTools.HeppyCore.framework.looper import Looper
from PhysicsTools.HeppyCore.framework.config import split

# global, to be used interactively when only one component is processed.
loop = None

def callBack( result ):
    pass

def runLoopAsync(comp, outDir, configName, options):
    try:
        loop = runLoop( comp, outDir, copy.copy(sys.modules[configName].config), options)
        return loop.name
    except Exception:
        import traceback
        print "ERROR processing component %s" % comp.name
        print comp
        print "STACK TRACE: "
        print traceback.format_exc()
        raise

_globalGracefulStopFlag = multiprocessing.Value('i',0)
def runLoop( comp, outDir, config, options):
    fullName = '/'.join( [outDir, comp.name ] )
    # import pdb; pdb.set_trace()
    config.components = [comp]
    memcheck = 2 if getattr(options,'memCheck',False) else -1
    loop = Looper( fullName,
                   config,
                   options.nevents, 0,
                   nPrint = options.nprint,
                   timeReport = options.timeReport,
                   quiet = options.quiet,
                   memCheckFromEvent = memcheck,
                   stopFlag = _globalGracefulStopFlag)
    # print loop
    if options.iEvent is None:
        loop.loop()
        loop.write()
        # print loop
    else:
        # loop.InitOutput()
        iEvent = int(options.iEvent)
        loop.process( iEvent )
    return loop


def createOutputDir(dirname, components, force):
    '''Creates the output dir, dealing with the case where dir exists.'''
    answer = None
    try:
        os.mkdir(dirname)
        return True
    except OSError:
        if not os.listdir(dirname):
            return True 
        else: 
            if force is True:
                return True
            else: 
                print 'directory %s already exists' % dirname
                print 'contents: '
                dirlist = [path for path in os.listdir(dirname) \
                               if os.path.isdir( '/'.join([dirname, path]) )]
                pprint( dirlist )
                print 'component list: '
                print [comp.name for comp in components]
                while answer not in ['Y','y','yes','N','n','no']:
                    answer = raw_input('Continue? [y/n]')
                if answer.lower().startswith('n'):
                    return False
                elif answer.lower().startswith('y'):
                    return True
                else:
                    raise ValueError( ' '.join(['answer can not have this value!',
                                                answer]) )
            

_heppyGlobalOptions = {}

def getHeppyOption(name,default=None):
    global _heppyGlobalOptions
    return _heppyGlobalOptions[name] if name in _heppyGlobalOptions else default
def setHeppyOption(name,value=True):
    global _heppyGlobalOptions
    _heppyGlobalOptions[name] = value

def main( options, args, parser ):

    if len(args) != 2:
        parser.print_help()
        print 'ERROR: please provide the processing name and the component list'
        sys.exit(1)

    outDir = args[0]
    if os.path.exists(outDir) and not os.path.isdir( outDir ):
        parser.print_help()
        print 'ERROR: when it exists, first argument must be a directory.'
        sys.exit(2)
    cfgFileName = args[1]
    if not os.path.isfile( cfgFileName ):
        parser.print_help()
        print 'ERROR: second argument must be an existing file (your input cfg).'
        sys.exit(3)

    if options.verbose:
        import logging
        logging.basicConfig(level=logging.INFO)

    # Propagate global options to _heppyGlobalOptions within this module
    # I have to import it explicitly, 'global' does not work since the
    # module is not set when executing the main
    from PhysicsTools.HeppyCore.framework.heppy_loop import _heppyGlobalOptions
    for opt in options.extraOptions:
        if "=" in opt:
            (key,val) = opt.split("=",1)
            _heppyGlobalOptions[key] = val
        else:
            _heppyGlobalOptions[opt] = True

    file = open( cfgFileName, 'r' )
    sys.path.append( os.path.dirname(cfgFileName) )
    cfg = imp.load_source( 'PhysicsTools.HeppyCore.__cfg_to_run__', 
                           cfgFileName, file)

    selComps = [comp for comp in cfg.config.components if len(comp.files)>0]
    selComps = split(selComps)
    # for comp in selComps:
    #    print comp
    if len(selComps)>options.ntasks:
        print "WARNING: too many threads {tnum}, will just use a maximum of {jnum}.".format(tnum=len(selComps),jnum=options.ntasks)
    if not createOutputDir(outDir, selComps, options.force):
        print 'exiting'
        sys.exit(0)
    if len(selComps)>1:
        shutil.copy( cfgFileName, outDir )
        pool = multiprocessing.Pool(processes=min(len(selComps),options.ntasks))
        ## workaround for a scoping problem in ipython+multiprocessing
        import PhysicsTools.HeppyCore.framework.heppy_loop as ML 
        for comp in selComps:
            pool.apply_async( ML.runLoopAsync, [comp, outDir, 'PhysicsTools.HeppyCore.__cfg_to_run__', options],
                              callback=ML.callBack)
        pool.close()
        pool.join()
    else:
        # when running only one loop, do not use multiprocessor module.
        # then, the exceptions are visible -> use only one sample for testing
        global loop
        loop = runLoop( comp, outDir, cfg.config, options )
    return loop


def create_parser(): 
    from optparse import OptionParser

    parser = OptionParser()
    parser.usage = """
    %prog <output_directory> <config_file>
    Start the processing of the jobs defined in your configuration file.
    """
    parser.add_option("-N", "--nevents",
                      dest="nevents",
                      type="int",
                      help="number of events to process",
                      default=None)
    parser.add_option("-p", "--nprint",
                      dest="nprint",
                      help="number of events to print at the beginning",
                      default=5)
    parser.add_option("-e", "--iEvent", 
                      dest="iEvent",
                      help="jump to a given event. ignored in multiprocessing.",
                      default=None)
    parser.add_option("-f", "--force",
                      dest="force",
                      action='store_true',
                      help="don't ask questions in case output directory already exists.",
                      default=False)
    parser.add_option("-i", "--interactive", 
                      dest="interactive",
                      action='store_true',
                      help="stay in the command line prompt instead of exiting",
                      default=False)
    parser.add_option("-t", "--timereport", 
                      dest="timeReport",
                      action='store_true',
                      help="Make a report of the time used by each analyzer",
                      default=False)
    parser.add_option("-v", "--verbose",
                      dest="verbose",
                      action='store_true',
                      help="increase the verbosity of the output (from 'warning' to 'info' level)",
                      default=False)
    parser.add_option("-q", "--quiet",
                      dest="quiet",
                      action='store_true',
                      help="do not print log messages to screen.",
                      default=False)
    parser.add_option("-o", "--option",
                      dest="extraOptions",
                      type="string",
                      action="append",
                      default=[],
                      help="Save one extra option (either a flag, or a key=value pair) that can be then accessed from the job config file")
    parser.add_option("-j", "--ntasks",
                      dest="ntasks",
                      type="int",
                      help="number of parallel tasks to span",
                      default=10)
    parser.add_option("--memcheck", 
                      dest="memCheck",
                      action='store_true',
                      help="Activate memory checks per event",
                      default=False)

    return parser
