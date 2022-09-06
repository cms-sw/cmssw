#!/usr/bin/env python3
from __future__ import print_function
import sys, os

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

    # test for wrong input workflows
    if opt.testList:
        definedWf = [dwf.numId for dwf in mrd.workFlows]
        definedSet = set(definedWf)
        testSet = set(opt.testList)
        undefSet = testSet - definedSet
        if len(undefSet)>0: raise ValueError('Undefined workflows: '+', '.join(map(str,list(undefSet))))
        duplicates = [wf for wf in testSet if definedWf.count(wf)>1 ]
        if len(duplicates)>0: raise ValueError('Duplicated workflows: '+', '.join(map(str,list(duplicates))))

    ret = 0
    if opt.show:
        mrd.show(opt.testList, opt.extended, opt.cafVeto)
        if opt.testList : print('testListected items:', opt.testList)
    else:
        mRunnerHi = MatrixRunner(mrd.workFlows, opt.nProcs, opt.nThreads)
        ret = mRunnerHi.runTests(opt)

    if opt.wmcontrol:
        if ret!=0:
            print('Cannot go on with wmagent injection with failing workflows')
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
                     7.3, #CosmicsSPLoose_UP17
                     8, #BH/Cosmic MC
                     25, #MC ttbar
                     4.22, #cosmic data
                     4.53, #run1 data + miniAOD
                     9.0, #Higgs200 charged taus
                     1000, #data+prompt
                     1001, #data+express
                     101.0, #SingleElectron120E120EHCAL
                     136.731, #2016B Photon data
                     136.7611, #2016E JetHT reMINIAOD from 80X legacy
                     136.8311, #2017F JetHT reMINIAOD from 94X reprocessing
                     136.88811,#2018D JetHT reMINIAOD from UL processing
                     136.793, #2017C DoubleEG
                     136.874, #2018C EGamma
                     138.4, #2021 MinimumBias prompt reco
                     138.5, #2021 MinimumBias express
                     139.001, #2021 MinimumBias offline with HLT step
                     140.53, #2011 HI data
                     140.56, #2018 HI data
                     158.01, #reMiniAOD of 2018 HI MC with pp-like reco
                     312.0, #2021/Run3 HI MC Pyquen_ZeemumuJets_pt10 with pp-like reco
                     1306.0, #SingleMu Pt1 UP15
                     1325.81, #test NanoAOD from existing MINI UL 106Xv1
                     136.8523, #test NanoAOD from existing reMINI UL 106Xv2
                     1330, #Run2 2015/2016 MC Zmm
                     135.4, #Run 2 2015/2016 Zee ttbar fastsim
                     10042.0, #2017 ZMM
                     10024.0, #2017 ttbar
                     10824.0, #2018 ttbar
                     2018.1, #2018 ttbar fastsim
                     11634.911, #2021 DD4hep ttbar reading geometry from XML
                     11634.914, #2021 DDD ttbar reading geometry from the DB
                     11634.0, #2021 ttbar (switching to DD4hep by default)
                     11634.7, #2021 ttbar mkFit
                     11834.0, #2021 ttbar PU
                     13234.0, #2021 ttbar fastsim
                     13434.0, #2021 ttbar PU fastsim
                     12434.0, #2023 ttbar
                     23234.0, #2026D49 ttbar (HLT TDR baseline w/ HGCal v11)
                     28234.0, #2026D60 (exercise HF nose)
                     35034.0, #2026D77 ttbar
                     39434.0, #2026D88 ttbar (2022 new baseline)
                     39434.75, #2026D88 ttbar with HLT75e33
                     #39434.911, #2026D88 ttbar DD4hep XML
                     39634.999, #2026D88 ttbar premixing stage1+stage2, PU50
                     39496.0, #CE_E_Front_120um D88
                     39500.0, #CE_H_Coarse_Scint D88 
                     25202.0, #2016 ttbar UP15 PU
                     250202.181, #2018 ttbar stage1 + stage2 premix
                     ],
        'jetmc': [5.1, 13, 15, 25, 38, 39], #MC
        'metmc' : [5.1, 15, 25, 37, 38, 39], #MC
        'muonmc' : [5.1, 124.4, 124.5, 20, 21, 22, 23, 25, 30], #MC
        }


    import argparse
    usage = 'usage: runTheMatrix.py --show -s '

    parser = argparse.ArgumentParser(usage,formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-b','--batchName',
                        help='relval batch: suffix to be appended to Campaign name',
                        dest='batchName',
                        default='')

    parser.add_argument('-m','--memoryOffset',
                        help='memory of the wf for single core',
                        dest='memoryOffset',
                        type=int,
                        default=3000)

    parser.add_argument('--addMemPerCore',
                        help='increase of memory per each n > 1 core:  memory(n_core) = memoryOffset + (n_core-1) * memPerCore',
                        dest='memPerCore',
                        type=int,
                        default=1500)
    
    parser.add_argument('-j','--nproc',
                        help='number of processes. 0 Will use 4 processes, not execute anything but create the wfs',
                        dest='nProcs',
                        type=int,
                        default=4)
    
    parser.add_argument('-t','--nThreads',
                        help='number of threads per process to use in cmsRun.',
                        dest='nThreads',
                        type=int,
                        default=1)
    
    parser.add_argument('--nStreams',
                        help='number of streams to use in cmsRun.',
                        dest='nStreams',
                        type=int,
                        default=0)
    
    parser.add_argument('--numberEventsInLuminosityBlock',
                        help='number of events in a luminosity block',
                        dest='numberEventsInLuminosityBlock',
                        type=int,
                        default=-1)

    parser.add_argument('-n','--showMatrix',
                        help='Only show the worflows. Use --ext to show more',
                        dest='show',
                        default=False,
                        action='store_true')
    
    parser.add_argument('-e','--extended',
                        help='Show details of workflows, used with --show',
                        dest='extended',
                        default=False,
                        action='store_true')
    
    parser.add_argument('-s','--selected',
                        help='Run a pre-defined selected matrix of wf. Deprecated, please use -l limited',
                        dest='restricted',
                        default=False,
                        action='store_true')
    
    parser.add_argument('-l','--list',
                        help='Comma separated list of workflow to be shown or ran. Possible keys are also '+str(predefinedSet.keys())+'. and wild card like muon, or mc',
                        dest='testList',
                        default=None)
    
    parser.add_argument('-r','--raw',
                        help='Temporary dump the .txt needed for prodAgent interface. To be discontinued soon. Argument must be the name of the set (standard, pileup,...)',
                        dest='raw')
    
    parser.add_argument('-i','--useInput',
                        help='Use recyling where available. Either all, or a comma separated list of wf number.',
                        dest='useInput',
                        type=lambda x: x.split(','),
                        default=None)
    
    parser.add_argument('-w','--what',
                        help='Specify the set to be used. Argument must be the name of a set (standard, pileup,...) or multiple sets separated by commas (--what standard,pileup )',
                        dest='what',
                        default='all')
    
    parser.add_argument('--step1',
                        help='Used with --raw. Limit the production to step1',
                        dest='step1Only',
                        default=False)
    
    parser.add_argument('--maxSteps',
                        help='Only run maximum on maxSteps. Used when we are only interested in first n steps.',
                        dest='maxSteps',
                        default=9999,
                        type=int)
    
    parser.add_argument('--fromScratch',
                        help='Comma separated list of wf to be run without recycling. all is not supported as default.',
                        dest='fromScratch',
                        type=lambda x: x.split(','),
                        default=None)
    
    parser.add_argument('--refRelease',
                        help='Allow to modify the recycling dataset version',
                        dest='refRel',
                        default=None)
    
    parser.add_argument('--wmcontrol',
                        help='Create the workflows for injection to WMAgent. In the WORKING. -wmcontrol init will create the the workflows, -wmcontrol test will dryRun a test, -wmcontrol submit will submit to wmagent',
                        choices=['init','test','submit','force'],
                        dest='wmcontrol',
                        default=None)
    
    parser.add_argument('--revertDqmio',
                        help='When submitting workflows to wmcontrol, force DQM outout to use pool and not DQMIO',
                        choices=['yes','no'],
                        dest='revertDqmio',
                        default='no')
    
    parser.add_argument('--optionswm',
                        help='Specify a few things for wm injection',
                        default='',
                        dest='wmoptions')
    
    parser.add_argument('--keep',
                        help='allow to specify for which comma separated steps the output is needed',
                        default=None)
    
    parser.add_argument('--label',
                        help='allow to give a special label to the output dataset name',
                        default='')
    
    parser.add_argument('--command',
                        help='provide a way to add additional command to all of the cmsDriver commands in the matrix',
                        dest='command',
                        action='append',
                        default=None)
    
    parser.add_argument('--apply',
                        help='allow to use the --command only for 1 comma separeated',
                        dest='apply',
                        default=None)
    
    parser.add_argument('--workflow',
                        help='define a workflow to be created or altered from the matrix',
                        action='append',
                        dest='workflow',
                        default=None)
    
    parser.add_argument('--dryRun',
                        help='do not run the wf at all',
                        action='store_true',
                        dest='dryRun',
                        default=False)
    
    parser.add_argument('--testbed',
                        help='workflow injection to cmswebtest (you need dedicated rqmgr account)',
                        dest='testbed',
                        default=False,
                        action='store_true')
    
    parser.add_argument('--noCafVeto',
                        help='Run from any source, ignoring the CAF label',
                        dest='cafVeto',
                        default=True,
                        action='store_false')
    
    parser.add_argument('--overWrite',
                        help='Change the content of a step for another. List of pairs.',
                        dest='overWrite',
                        default=None)
    
    parser.add_argument('--noRun',
                        help='Remove all run list selection from wfs',
                        dest='noRun',
                        default=False,
                        action='store_true')

    parser.add_argument('--das-options',
                        help='Options to be passed to dasgoclient.',
                        dest='dasOptions',
                        default="--limit 0",
                        action='store')

    parser.add_argument('--job-reports',
                        help='Dump framework job reports',
                        dest='jobReports',
                        default=False,
                        action='store_true')
    
    parser.add_argument('--ibeos',
                        help='Use IB EOS site configuration',
                        dest='IBEos',
                        default=False,
                        action='store_true')
    
    parser.add_argument('--sites',
                        help='Run DAS query to get data from a specific site. Set it to empty string to search all sites.',
                        dest='dasSites',
                        default='T2_CH_CERN',
                        action='store')
    
    parser.add_argument('--interactive',
                        help="Open the Matrix interactive shell",
                        action='store_true',
                        default=False)
    
    parser.add_argument('--dbs-url',
                        help='Overwrite DbsUrl value in JSON submitted to ReqMgr2',
                        dest='dbsUrl',
                        default=None,
                        action='store')
    
    gpugroup = parser.add_argument_group('GPU-related options','These options are only meaningful when --gpu is used, and is not set to forbidden.')

    gpugroup.add_argument('--gpu','--requires-gpu',
                          help='Enable GPU workflows. Possible options are "forbidden" (default), "required" (implied if no argument is given), or "optional".',
                          dest='gpu',
                          choices=['forbidden', 'optional', 'required'],
                          nargs='?',
                          const='required',
                          default='forbidden',
                          action='store')

    gpugroup.add_argument('--gpu-memory',
                          help='Specify the minimum amount of GPU memory required by the job, in MB.',
                          dest='GPUMemoryMB',
                          type=int,
                          default=8000)
    
    gpugroup.add_argument('--cuda-capabilities',
                          help='Specify a comma-separated list of CUDA "compute capabilities", or GPU hardware architectures, that the job can use.',
                          dest='CUDACapabilities',
                          type=lambda x: x.split(','),
                          default='6.0,6.1,6.2,7.0,7.2,7.5,8.0,8.6')
    
    # read the CUDA runtime version included in CMSSW
    cudart_version = None
    libcudart = os.path.realpath(os.path.expandvars('$CMSSW_RELEASE_BASE/external/$SCRAM_ARCH/lib/libcudart.so'))
    if os.path.isfile(libcudart):
        cudart_basename = os.path.basename(libcudart)
        cudart_version = '.'.join(cudart_basename.split('.')[2:4])
    gpugroup.add_argument('--cuda-runtime',
                          help='Specify major and minor version of the CUDA runtime used to build the application.',
                          dest='CUDARuntime',
                          default=cudart_version)
    
    gpugroup.add_argument('--force-gpu-name',
                          help='Request a specific GPU model, e.g. "Tesla T4" or "NVIDIA GeForce RTX 2080". The default behaviour is to accept any supported GPU.',
                          dest='GPUName',
                          default='')
    
    gpugroup.add_argument('--force-cuda-driver-version',
                          help='Request a specific CUDA driver version, e.g. 470.57.02. The default behaviour is to accept any supported CUDA driver version.',
                          dest='CUDADriverVersion',
                          default='')
    
    gpugroup.add_argument('--force-cuda-runtime-version',
                          help='Request a specific CUDA runtime version, e.g. 11.4. The default behaviour is to accept any supported CUDA runtime version.',
                          dest='CUDARuntimeVersion',
                          default='')
    
    opt = parser.parse_args()
    if opt.command: opt.command = ' '.join(opt.command)
    os.environ["CMSSW_DAS_QUERY_SITES"]=opt.dasSites
    if opt.IBEos:
      from subprocess import getstatusoutput as run_cmd

      ibeos_cache = os.path.join(os.getenv("LOCALRT"), "ibeos_cache.txt")
      if not os.path.exists(ibeos_cache):
        err, out = run_cmd("curl -L -s -o %s https://raw.githubusercontent.com/cms-sw/cms-sw.github.io/master/das_queries/ibeos.txt" % ibeos_cache)
        if err:
          run_cmd("rm -f %s" % ibeos_cache)
          print("Error: Unable to download ibeos cache information")
          print(out)
          sys.exit(err)

      for cmssw_env in [ "CMSSW_BASE", "CMSSW_RELEASE_BASE" ]:
        cmssw_base = os.getenv(cmssw_env,None)
        if not cmssw_base: continue
        cmssw_base = os.path.join(cmssw_base,"src/Utilities/General/ibeos")
        if os.path.exists(cmssw_base):
          os.environ["PATH"]=cmssw_base+":"+os.getenv("PATH")
          os.environ["CMS_PATH"]="/cvmfs/cms-ib.cern.ch"
          os.environ["SITECONFIG_PATH"]="/cvmfs/cms-ib.cern.ch/SITECONF/local"
          os.environ["CMSSW_USE_IBEOS"]="true"
          print(">> WARNING: You are using SITECONF from /cvmfs/cms-ib.cern.ch")
          break
    if opt.restricted:
        print('Deprecated, please use -l limited')
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
                    print(entry,'is not a possible selected entry')

        opt.testList = list(set(testList))
    
    if opt.wmcontrol:
        performInjectionOptionTest(opt)
    if opt.overWrite:
        opt.overWrite=eval(opt.overWrite)
    if opt.interactive:
        import cmd

        class TheMatrix(cmd.Cmd):
            intro = "Welcome to the Matrix (? for help)"
            prompt = "matrix> "

            def __init__(self, opt):
                cmd.Cmd.__init__(self)
                self.opt_ = opt
                self.matrices_ = {}
                tmp = MatrixReader(self.opt_)
                for what in tmp.files:
                    what = what.replace('relval_','')
                    self.opt_.what = what
                    self.matrices_[what] = MatrixReader(self.opt_)
                    self.matrices_[what].prepare(self.opt_.useInput, self.opt_.refRel,
                                                self.opt_.fromScratch)
                os.system("clear")

            def do_clear(self, arg):
                """Clear the screen, put prompt at the top"""
                os.system("clear")

            def do_exit(self, arg):
                print("Leaving the Matrix")
                return True

            def default(self, inp):
                if inp == 'x' or inp == 'q':
                    return self.do_exit(inp)

            def help_predefined(self):
                print("\n".join(["predefined [predef1 [...]]\n",
                "Run w/o argument, it will print the list of known predefined workflows.",
                "Run with space-separated predefined workflows, it will print the workflow-ids registered to them"]))

            def complete_predefined(self, text, line, start_idx, end_idx):
                if text and len(text) > 0:
                    return [t for t in predefinedSet.keys() if t.startswith(text)]
                else:
                    return predefinedSet.keys()

            def do_predefined(self, arg):
                """Print the list of predefined workflows"""
                print("List of predefined workflows")
                if arg:
                    for w in arg.split():
                        if w in predefinedSet.keys():
                            print("Predefined Set: %s" % w)
                            print(predefinedSet[w])
                        else:
                            print("Unknown Set: %s" % w)
                else:
                    print(predefinedSet.keys())

            def help_showWorkflow(self):
                print("\n".join(["showWorkflow [workflow1 [...]]\n",
                    "Run w/o arguments, it will print the list of registered macro-workflows.",
                    "Run with space-separated workflows, it will print the full list of workflow-ids registered to them"]))

            def complete_showWorkflow(self, text, line, start_idx, end_idx):
                if text and len(text) > 0:
                    return [t for t in self.matrices_.keys() if t.startswith(text)]
                else:
                    return self.matrices_.keys()

            def do_showWorkflow(self, arg):
                if arg == '':
                    print("Available workflows:")
                    for k in self.matrices_.keys():
                        print(k)
                else:
                    selected = arg.split()
                    for k in selected:
                        if k not in self.matrices_.keys():
                            print("Unknown workflow %s: skipping" % k)
                        else:
                            for wfl in self.matrices_[k].workFlows:
                                wfName, stepNames = wfl.nameId.split('+',1)
                                print("%s %s %s" % (wfl.numId, wfName, stepNames))
                            print("%s contains %d workflows" % (k, len(self.matrices_[k].workFlows)))

            def help_searchInWorkflow(self):
                print("\n".join(["searchInWorkflow wfl_name search_regexp\n",
                    "This command will search for a match within all workflows registered to wfl_name.",
                    "The search is done on both the workflow name and the names of steps registered to it."]))

            def complete_searchInWorkflow(self, text, line, start_idx, end_idx):
                if text and len(text) > 0:
                    return [t for t in self.matrices_.keys() if t.startswith(text)]
                else:
                    return self.matrices_.keys()

            def do_searchInWorkflow(self, arg):
                args = arg.split()
                if len(args) < 2:
                    print("searchInWorkflow name regexp")
                    return
                if args[0] not in self.matrices_.keys():
                    print("Unknown workflow")
                    return
                import re
                pattern = None
                try:
                    pattern = re.compile(args[1])
                except:
                    print("Failed to compile regexp %s" % args[1])
                    return
                counter = 0
                for wfl in self.matrices_[args[0]].workFlows:
                    wfName, stepNames = wfl.nameId.split('+',1)
                    if re.match(pattern, wfName) or re.match(pattern, stepNames):
                        print("%s %s %s" % (wfl.numId, wfName, stepNames))
                        counter += 1
                print("Found %d compatible workflows inside %s" % (counter, args[0]))

            def help_search(self):
                print("\n".join(["search search_regexp\n",
                    "This command will search for a match within all workflows registered.",
                    "The search is done on both the workflow name and the names of steps registered to it."]))

            def do_search(self, arg):
                args = arg.split()
                if len(args) < 1:
                    print("search regexp")
                    return
                for wfl in self.matrices_.keys():
                    self.do_searchInWorkflow(' '.join([wfl, args[0]]))

            def help_dumpWorkflowId(self):
                print("\n".join(["dumpWorkflowId [wfl-id1 [...]]\n",
                    "Dumps the details (cmsDriver commands for all steps) of the space-separated workflow-ids in input."]))

            def do_dumpWorkflowId(self, arg):
                wflids = arg.split()
                if len(wflids) == 0:
                    print("dumpWorkflowId [wfl-id1 [...]]")
                    return

                fmt   = "[%d]: %s\n"
                maxLen = 100
                for wflid in wflids:
                    dump = True
                    for key, mrd in self.matrices_.items():
                        for wfl in mrd.workFlows:
                            if wfl.numId == float(wflid):
                                wfName, stepNames = wfl.nameId.split('+',1)
                                if dump:
                                    dump = False
                                    print(wfl.numId, stepNames)
                                    for i,s in enumerate(wfl.cmds):
                                        print(fmt % (i+1, (str(s)+' ')))
                                    print("\nWorkflow found in %s." % key)
                                else:
                                    print("Workflow also found in %s." % key)

            do_EOF = do_exit

        TheMatrix(opt).cmdloop()
        sys.exit(0)

    if opt.raw and opt.show: ###prodAgent to be discontinued
        ret = showRaw(opt)
    else:
        ret = runSelected(opt)


    sys.exit(ret)
