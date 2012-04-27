#!/usr/bin/env python

import sys

from Configuration.PyReleaseValidation.MatrixReader import MatrixReader
from Configuration.PyReleaseValidation.MatrixRunner import MatrixRunner
from Configuration.PyReleaseValidation.MatrixInjector import MatrixInjector,performInjectionOptionTest
        
# ================================================================================

def showRaw(opt):

    mrd = MatrixReader(opt)
    mrd.showRaw(opt.useInput, opt.refRel, opt.fromScratch, opt.raw, opt.step1Only, selected=opt.testList)

    return 0
        
# ================================================================================

def runSelected(opt):

    mrd = MatrixReader(opt)
    mrd.prepare(opt.useInput, opt.refRel, opt.fromScratch)

    ret = 0
    if opt.show:
        mrd.show(opt.testList,opt.extended)
        if opt.testList : print 'testListected items:', opt.testList
    else:
        mRunnerHi = MatrixRunner(mrd.workFlows, opt.nThreads)
        ret = mRunnerHi.runTests(opt.testList,opt.dryRun)

    if opt.wmcontrol:
        if ret!=0:
            print 'Cannot go on with wmagent injection with failing workflows'
        else:
            wfInjector = MatrixInjector(mode=opt.wmcontrol)
            ret= wfInjector.prepare(mrd,
                                    mRunnerHi.runDirs)
            if ret==0:
                wfInjector.upload()
                wfInjector.submit()
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
                      default=None
                      )
    parser.add_option('--wmcontrol',
                      help='Create the workflows for injection to WMAgent. In the WORKING. -wmcontrol init will create the the workflows, -wmcontrol test will dryRun a test, -wmcontrol submit will submit to wmagent',
                      choices=['init','test','submit','force'],
                      dest='wmcontrol',
                      default=None,
                      )
    parser.add_option('--command',
                      help='provide a way to add additional command to all of the cmsDriver commands in the matrix',
                      dest='command',
                      default=None
                      )
    parser.add_option('--workflow',
                      help='define a workflow to be created or altered from the matrix',
                      action='append',
                      dest='workflow',
                      default=None
                      )
    parser.add_option('--dryRun',
                      help='do not run the wf at all',
                      action='store_true',
                      dest='dryRun',
                      default=False
                      )
    
    
    opt,args = parser.parse_args()
    if opt.testList: opt.testList = map(float,opt.testList.split(','))
    if opt.restricted:
        limitedMatrix=[5.1, #FastSim ttbar
                       8, #BH/Cosmic MC
                       25, #MC ttbar
                       4.22, #cosmic data
                       4.291, #hlt data
                       1000, #data+prompt
                       1001, #data+express
                       4.53, #HI data
                       40, #HI MC
                       ]
        if opt.testList:
            opt.testList.extend(limitedMatrix)
        else:
            opt.testList=limitedMatrix
    if opt.useInput: opt.useInput = opt.useInput.split(',')
    if opt.fromScratch: opt.fromScratch = opt.fromScratch.split(',')
    if opt.nThreads: opt.nThreads=int(opt.nThreads)

    if opt.wmcontrol:
        performInjectionOptionTest(opt)
        
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
