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
        ret = mRunnerHi.runTests(opt)

    if opt.wmcontrol:
        if ret!=0:
            print 'Cannot go on with wmagent injection with failing workflows'
        else:
            wfInjector = MatrixInjector(opt,mode=opt.wmcontrol,options=opt.wmoptions)
            ret= wfInjector.prepare(mrd,
                                    mRunnerHi.runDirs)
            if ret==0:
                wfInjector.upload()
                wfInjector.submit()
    return ret

# ================================================================================

if __name__ == '__main__':

    #this can get out of here
    predefinedSet={
        'limited' : [5.1, #FastSim ttbar
                     8, #BH/Cosmic MC
                     25, #MC ttbar
                     4.22, #cosmic data
                     4.291, #hlt data
                     1000, #data+prompt
                     1001, #data+express
                     4.53, #HI data
                     40, #HI MC
                     ],
        'jetmc': [5.1, 13, 15, 25, 38, 39], #MC
        'metmc' : [5.1, 15, 25, 37, 38, 39], #MC
        'muonmc' : [5.1, 124.4, 124.5, 20, 21, 22, 23, 25, 30], #MC
        }
        

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
                      help='Run a pre-defined selected matrix of wf. Deprecated, please use -l limited',
                      dest='restricted',
                      default=False,
                      action='store_true'
                      )
    parser.add_option('-l','--list',
                     help='Coma separated list of workflow to be shown or ran. Possible keys are also '+str(predefinedSet.keys())+'. and wild card like muon, or mc',
                     dest='testList',
                     default=None
                     )
    parser.add_option('-r','--raw',
                      help='Temporary dump the .txt needed for prodAgent interface. To be discontinued soon. Argument must be the name of the set (standard, pileup,...)',
                      dest='raw'
                      )
    parser.add_option('-i','--useInput',
                      help='Use recyling where available. Either all, or a coma separated list of wf number.',
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
                      help='Coma separated list of wf to be run without recycling. all is not supported as default.',
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
    parser.add_option('--revertDqmio',
                      help='When submitting workflows to wmcontrol, force DQM outout to use pool and not DQMIO',
                      choices=['yes','no'],
                      dest='revertDqmio',
                      default='no',
                      )
    parser.add_option('--optionswm',
                      help='Specify a few things for wm injection',
                      default='',
                      dest='wmoptions')
    parser.add_option('--keep',
                      help='allow to specify for which coma separated steps the output is needed',
                      default=None)
    parser.add_option('--label',
                      help='allow to give a special label to the output dataset name',
                      default='')
    parser.add_option('--command',
                      help='provide a way to add additional command to all of the cmsDriver commands in the matrix',
                      dest='command',
                      default=None
                      )
    parser.add_option('--apply',
                      help='allow to use the --command only for 1 coma separeated',
                      dest='apply',
                      default=None)
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
    parser.add_option('--noCafVeto',
                      help='Run from any source, ignoring the CAF label',
                      dest='cafVeto',
                      default=True,
                      action='store_false'
                      )
    parser.add_option('--overWrite',
                      help='Change the content of a step for another. List of pairs.',
                      dest='overWrite',
                      default=None
                      )
    parser.add_option('--noRun',
                      help='Remove all run list selection from wfs',
                      dest='noRun',
                      default=False,
                      action='store_true')

    parser.add_option('--das-options',
                      help='Options to be passed to das_client.py.',
                      dest='dasOptions',
                      default="--limit 0",
                      action='store')

    parser.add_option('--job-reports',
                      help='Dump framework job reports',
                      dest='jobReports',
                      default=False,
                      action='store_true')
    
    opt,args = parser.parse_args()
    if opt.restricted:
        print 'Deprecated, please use -l limited'
        if opt.testList:            opt.testList+=',limited'
        else:            opt.testList='limited'

    def stepOrIndex(s):
        if s.isdigit():
            return int(s)
        else:
            return s
    if opt.apply:
        opt.apply=map(stepOrIndex,opt.apply.split(','))
    if opt.keep:
        opt.keep=map(stepOrIndex,opt.keep.split(','))
        
                
                
    if opt.testList:
        testList=[]
        for entry in opt.testList.split(','):
            if not entry: continue
            mapped=False
            for k in predefinedSet:
                if k.lower().startswith(entry.lower()) or k.lower().endswith(entry.lower()):
                    testList.extend(predefinedSet[k])
                    mapped=True
                    break
            if not mapped:
                try:
                    testList.append(float(entry))
                except:
                    print entry,'is not a possible selected entry'
            
        opt.testList = list(set(testList))


    if opt.useInput: opt.useInput = opt.useInput.split(',')
    if opt.fromScratch: opt.fromScratch = opt.fromScratch.split(',')
    if opt.nThreads: opt.nThreads=int(opt.nThreads)

    if opt.wmcontrol:
        performInjectionOptionTest(opt)
    if opt.overWrite:
        opt.overWrite=eval(opt.overWrite)

    if opt.raw and opt.show: ###prodAgent to be discontinued
        ret = showRaw(opt)
    else:
        ret = runSelected(opt)


    sys.exit(ret)
