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

from CMGTools.RootTools.fwlite.Looper import Looper
from CMGTools.RootTools.fwlite.PythonPath import pythonpath

# global, to be used interactively when only one component is processed.
loop = None 

def callBack( result ):
    pass
    print 'production done:', str(result)

def runLoopAsync(comp, outDir, config, options):
    loop = runLoop( comp, outDir, config, options)
    return loop.name

def runLoop( comp, outDir, config, options):
    fullName = '/'.join( [outDir, comp.name ] )
    # import pdb; pdb.set_trace()
    loop = Looper( fullName, comp, config.sequence,
                   options.nevents, 0, 
                   nPrint = options.nprint)
    print loop
    if options.iEvent is None:
        loop.loop()
        loop.write()
        print loop
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
        if hasattr( comp, 'splitFactor') and comp.splitFactor>1:
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



def main( options, args ):
    
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

    file = open( cfgFileName, 'r' )
    cfg = imp.load_source( 'cfg', cfgFileName, file)

    sys.path = pythonpath + sys.path
 
    selComps = [comp for comp in cfg.config.components if len(comp.files)>0]
    selComps = split(selComps)
    for comp in selComps:
        print comp
    if len(selComps)>10:
        print "WARNING: too many threads {tnum}, will just use a maximum of 10.".format(tnum=len(selComps))
    if not createOutputDir(outDir, selComps, options.force):
        print 'exiting'
        sys.exit(0)
    if len(selComps)>1:
        shutil.copy( cfgFileName, outDir )
        pool = Pool(processes=min(len(selComps),10))
        ## workaround for a scoping problem in ipython+multiprocessing
        import CMGTools.RootTools.fwlite.MultiLoop as ML 
        for comp in selComps:
            print 'submitting', comp.name
            pool.apply_async( ML.runLoopAsync, [comp, outDir, cfg.config, options],
                              callback=ML.callBack)     
        pool.close()
        pool.join()
    else:
        # when running only one loop, do not use multiprocessor module.
        # then, the exceptions are visible -> use only one sample for testing
        global loop
        loop = runLoop( comp, outDir, cfg.config, options )



if __name__ == '__main__':
    from optparse import OptionParser
    
    parser = OptionParser()
    parser.usage = """
    %prog <name> <analysis_cfg>
    For each component, start a Loop.
    'name' is whatever you want.
    """

    parser.add_option("-N", "--nevents", 
                      dest="nevents", 
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



    (options,args) = parser.parse_args()


    main(options, args)
    if not options.interactive:
        exit() 
