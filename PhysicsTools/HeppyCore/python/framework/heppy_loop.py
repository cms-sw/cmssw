#!/usr/bin/env python
# Copyright (C) 2014 Colin Bernet
# https://github.com/cbernet/heppy/blob/master/LICENSE

import os
import shutil
import glob
import sys
import imp
import copy
from multiprocessing import Pool
from pprint import pprint

# import root in batch mode if "-i" is not among the options
if "-i" not in sys.argv:
    oldv = sys.argv[:]
    sys.argv = [ "-b-"]
    import ROOT
    ROOT.gROOT.SetBatch(True)
    sys.argv = oldv


from PhysicsTools.HeppyCore.framework.looper import Looper

# global, to be used interactively when only one component is processed.
loop = None

def callBack( result ):
    pass
    print 'production done:', str(result)

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

def runLoop( comp, outDir, config, options):
    fullName = '/'.join( [outDir, comp.name ] )
    # import pdb; pdb.set_trace()
    config.components = [comp]
    loop = Looper( fullName,
                   config,
                   options.nevents, 0,
                   nPrint = options.nprint,
                   timeReport = options.timeReport,
                   quiet = options.quiet)
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


def createOutputDir(dir, components, force):
    '''Creates the output dir, dealing with the case where dir exists.'''
    answer = None
    try:
        os.mkdir(dir)
        return True
    except OSError:
        print 'directory %s already exists' % dir
        print 'contents: '
        dirlist = [path for path in os.listdir(dir) if os.path.isdir( '/'.join([dir, path]) )]
        pprint( dirlist )
        print 'component list: '
        print [comp.name for comp in components]
        if force is True:
            print 'force mode, continue.'
            return True
        else:
            while answer not in ['Y','y','yes','N','n','no']:
                answer = raw_input('Continue? [y/n]')
            if answer.lower().startswith('n'):
                return False
            elif answer.lower().startswith('y'):
                return True
            else:
                raise ValueError( ' '.join(['answer can not have this value!',
                                            answer]) )

def chunks(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]

def split(comps):
    # import pdb; pdb.set_trace()
    splitComps = []
    for comp in comps:
        if hasattr( comp, 'fineSplitFactor') and comp.fineSplitFactor>1:
            subchunks = range(comp.fineSplitFactor)
            for ichunk, chunk in enumerate([(f,i) for f in comp.files for i in subchunks]):
                newComp = copy.deepcopy(comp)
                newComp.files = [chunk[0]]
                newComp.fineSplit = ( chunk[1], comp.fineSplitFactor )
                newComp.name = '{name}_Chunk{index}'.format(name=newComp.name,
                                                       index=ichunk)
                splitComps.append( newComp )
        elif hasattr( comp, 'splitFactor') and comp.splitFactor>1:
            chunkSize = len(comp.files) / comp.splitFactor
            if len(comp.files) % comp.splitFactor:
                chunkSize += 1
            # print 'chunk size',chunkSize, len(comp.files), comp.splitFactor
            for ichunk, chunk in enumerate( chunks( comp.files, chunkSize)):
                newComp = copy.deepcopy(comp)
                newComp.files = chunk
                newComp.name = '{name}_Chunk{index}'.format(name=newComp.name,
                                                       index=ichunk)
                splitComps.append( newComp )
        else:
            splitComps.append( comp )
    return splitComps


_heppyGlobalOptions = {}

def getHeppyOption(name,default=None):
    global _heppyGlobalOptions
    return _heppyGlobalOptions[name] if name in _heppyGlobalOptions else default

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
    cfg = imp.load_source( 'PhysicsTools.HeppyCore.__cfg_to_run__', cfgFileName, file)

    selComps = [comp for comp in cfg.config.components if len(comp.files)>0]
    selComps = split(selComps)
    # for comp in selComps:
    #    print comp
    if len(selComps)>10:
        print "WARNING: too many threads {tnum}, will just use a maximum of 10.".format(tnum=len(selComps))
    if not createOutputDir(outDir, selComps, options.force):
        print 'exiting'
        sys.exit(0)
    if len(selComps)>1:
        shutil.copy( cfgFileName, outDir )
        pool = Pool(processes=min(len(selComps),10))
        ## workaround for a scoping problem in ipython+multiprocessing
        import PhysicsTools.HeppyCore.framework.heppy_loop as ML 
        for comp in selComps:
            print 'submitting', comp.name
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
