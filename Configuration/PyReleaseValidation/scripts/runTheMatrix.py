#!/usr/bin/env python

import sys

from Configuration.PyReleaseValidation.MatrixReader import MatrixReader
from Configuration.PyReleaseValidation.MatrixRunner import MatrixRunner
        
# ================================================================================

def showRaw(useInput=None, refRel='', fromScratch=None, what='standard',step1Only=False,testList=None) :

    mrd = MatrixReader()
    mrd.showRaw(useInput, refRel, fromScratch, what, step1Only, selected=testList)

    return 0
        
# ================================================================================

def runSelected(testList, nThreads=4, show=False, useInput=None, refRel='', fromScratch=None) :

    stdList = ['5.2', # SingleMu10 FastSim
               '7',   # Cosmics+RECOCOS+ALCACOS
               '8',   # BeamHalo+RECOCOS+ALCABH
               '25',  # TTbar+RECO2+ALCATT2  STARTUP
               ]
    hiStatList = [
                  '121',   # TTbar_Tauola
                  '123.3', # TTBar FastSim
                   ]

    mrd = MatrixReader(noRun=(nThreads==0))
    mrd.prepare(useInput, refRel, fromScratch)

    if testList == []:
        testList = [float(x) for x in stdList+hiStatList]

    ret = 0
    if show:
        mrd.show(testList)
        if testList : print 'selected items:', testList
    else:
        mRunnerHi = MatrixRunner(mrd.workFlows, nThreads)
        ret = mRunnerHi.runTests(testList)

    return ret

# --------------------------------------------------------------------------------


def usage():
    print "Usage:", sys.argv[0], ' [options] '
    print """
Where options is one of the following:
  -d, --data <list> comma-separated list of workflows to use from the realdata file.
                    <list> can be "all" to select all data workflows
  -l, --list <list> comma-separated list of workflows to use from the cmsDriver*.txt files
  -j, --nproc <n>   run <n> processes in parallel (default: 4 procs)
  -s, --selected    run a subset of 8 workflows (usually in the CustomIB)
  -n, -q, --show    show the (selected) workflows
  -i, --useInput <list>      will use data input (if defined) for the step1 instead of step1. <list> can be "all" for this option
      --refRelease <refRel>  will use <refRel> as reference release in datasets used for input (replacing the sim step)
  -r, --raw <what>  in combination with --show will create the old style cmsDriver_<what>_hlt.txt file (in the working dir)
  
<list>s should be put in single- or double-quotes to avoid confusion with/by the shell
"""

# ================================================================================

if __name__ == '__main__':

    import getopt
    
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hj:sl:nqo:d:i:r:", ['help',"nproc=",'selected','list=','showMatrix','only=','data=','useInput=','raw=', 'refRelease=','fromScratch=','step1','what='])
    except getopt.GetoptError, e:
        print "unknown option", str(e)
        sys.exit(2)
        
# check command line parameters

    # set this to None if you want fromScratch as default, set it to 'all' to set the default to from input files.
    useInput = None  # step1 default is cmsDriver (i.e. "from scratch")
    # useInput = 'all' # step1 default is reading from input files

    
    np=4 # default: four threads
    sel = None
    fromScratch = None
    show = False
    data = None
    raw  = None
    refRel = ''
    step1Only=False
    what = 'all'
    for opt, arg in opts :
        if opt in ('-h','--help'):
            usage()
            sys.exit(0)
        if opt in ('-j', "--nproc" ):
            np=int(arg)
        if opt in ('-n','-q','--showMatrix', ):
            show = True
        if opt in ('-s','--selected',) :
            sel = []
        if opt in ('-l','--list',) :
            sel = arg.split(',')
        if opt in ('--fromScratch',) :
            fromScratch = arg.split(',')
        if opt in ('-i','--useInput',) :
            useInput = arg.split(',')
        if opt in ('--refRelease',) :
            refRel = arg
        if opt in ('-d','--data',) :
            data = arg.split(',')
        if opt in ('-r','--raw') :
            raw = arg
        if opt in ('-w','--what'):
            what=arg
        if opt in ('--step1'):
            step1Only=True
            
    # some sanity checking:
    if useInput and useInput != 'all' :
        for item in useInput:
            if fromScratch and item in fromScratch:
                print "FATAL error: request to run workflow ",item,'from scratch and using input. '
                sys.exit(-1)
        
    if raw and show:
        ret = showRaw(useInput=useInput, refRel=refRel,fromScratch=fromScratch, what=raw, step1Only=step1Only,testList=sel)
        sys.exit(ret)

    ret = runSelected(testList=sel, nThreads=np, show=show, useInput=useInput, refRel=refRel,fromScratch=fromScratch)

    sys.exit(ret)
