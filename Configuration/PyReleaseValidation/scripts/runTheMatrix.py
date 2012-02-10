#!/usr/bin/env python

import sys

from Configuration.PyReleaseValidation.MatrixReader import MatrixReader
from Configuration.PyReleaseValidation.MatrixRunner import MatrixRunner
        
# ================================================================================

def showRaw(opt):

    mrd = MatrixReader()
    mrd.showRaw(opt.useInput, opt.refRel, opt.fromScratch, opt.what, opt.step1Only, selected=opt.testList)

    return 0
        
# ================================================================================

def runSelected(opt):

    stdList = ['5.2', # SingleMu10 FastSim
               '7',   # Cosmics+RECOCOS+ALCACOS
               '8',   # BeamHalo+RECOCOS+ALCABH
               '25',  # TTbar+RECO2+ALCATT2  STARTUP
               ]
    hiStatList = [
                  '121',   # TTbar_Tauola
                  '123.3', # TTBar FastSim
                   ]

    mrd = MatrixReader(noRun=(opt.nThreads==0),what=opt.what)
    mrd.prepare(opt.useInput, opt.refRel, opt.fromScratch)

    if opt.testList == []:
        opt.testList = [float(x) for x in stdList+hiStatList]

    ret = 0
    if opt.show:
        mrd.show(opt.testList,opt.extended)
        if opt.testList : print 'testListected items:', opt.testList
    else:
        mRunnerHi = MatrixRunner(mrd.workFlows, opt.nThreads)
        ret = mRunnerHi.runTests(opt.testList)

    return ret

# ================================================================================

if __name__ == '__main__':

    import optparse
    usage = 'usage: runTheMatrix.py --show -s '

    parser = optparse.OptionParser(usage)

    parser.add_option('-j','--nproc',
                      help='number of threads. 0 Will use 4 threads, not execute anything but create the wfs',
                      dest='nThreads',
                      default=4
                     )
    parser.add_option('-n','--showMatrix',
                      help='Only show the worflows. Use --ext to show more',
                      dest='show',
                      default=False,
                      action='store_true'
                      )
    parser.add_option('-e','--extended',
                      help='Show details of workflows, used with --show',
                      dest='extended',
                      default=False,
                      action='store_true'
                      )
    parser.add_option('-s','--selected',
                      help='Run a pre-defined selected matrix of wf',
                      dest='restricted',
                      default=False,
                      action='store_true'
                      )
    parser.add_option('-l','--list',
                     help='Coma separated list of workflow to be shown or ran',
                     dest='testList',
                     default=None
                     )
    parser.add_option('-r','--raw',
                      help='Temporary dump the .txt needed for prodAgent interface. To be discontinued soon. Argument must be the name of the set (standard, pileup,...)',
                      dest='raw'
                      )
    parser.add_option('-i','--useInput',
                      help='Use recyling where available',
                      dest='useInput',
                      default=None
                      )
    parser.add_option('-w','--what',
                      help='Specify the set to be used. Argument must be the name of the set (standard, pileup,...)',
                      dest='what',
                      default='all'
                      )
    parser.add_option('--step1',
                      help='Used with --raw. Limit the production to step1',
                      dest='step1Only',
                      default=False
                      )
    parser.add_option('--fromScratch',
                      help='Coma separated list of wf to be run without recycling',
                      dest='fromScratch',
                      default=None
                       )
    parser.add_option('--refRelease',
                      help='Allow to modify the recycling dataset version',
                      dest='refRel',
                      default=''
                      )
    
    opt,args = parser.parse_args()
    if opt.testList: opt.testList = opt.testList.split(',')
    if opt.restricted:
        if opt.testList: print 'going for selected matrix, -l option is concurrent and arguments are dropped'
        opt.testList=[]
    if opt.useInput: opt.useInput = opt.useInput.split(',')
    if opt.fromScratch: opt.fromScratch = opt.fromScratch.split(',')
                     
    
    # some sanity checking:
    if opt.useInput and opt.useInput != 'all' :
        for item in opt.useInput:
            if opt.fromScratch and item in opt.fromScratch:
                print 'FATAL error: request to run workflow ',item,'from scratch and using input. '
                sys.exit(-1)
        
    if opt.raw and opt.show: ###prodAgent to be discontinued
        ret = showRaw(opt)
    else:
        ret = runSelected(opt)

    sys.exit(ret)
