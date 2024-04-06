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
        if opt.testList : print('selected items:', opt.testList)
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
        'limited' : [
                    # See README for further details
                    ###### MC (generated from scratch or from RelVals)
                    ### FullSim
                    # Run1 
                    5.1,        # TTbar_8TeV_TuneCUETP8M1       FastSim                                 
                    8,          # RelValBeamHalo                Cosmics
                    9.0,        # RelValHiggs200ChargedTaus             
                    25,         # RelValTTbar                           
                    101.0,      # SingleElectronE120EHCAL       + ECALHCAL.customise + fullMixCustomize_cff.setCrossingFrameOn
                    
                    # Run2
                    7.3,        # UndergroundCosmicSPLooseMu            
                    1306.0,     # RelValSingleMuPt1_UP15                
                    1330,       # RelValZMM_13                          
                    135.4,      # ZEE_13TeV_TuneCUETP8M1                
                    25202.0,    # RelValTTbar_13                PU = AVE_35_BX_25ns
                    250202.181, # RelValTTbar_13                PREMIX   

                    # Run3
                    11634.0,    # TTbar_14TeV                   2021
                    13234.0,    # RelValTTbar_14TeV             2021 FastsSim
                    12434.0,    # RelValTTbar_14TeV             2023
                    12834.0,    # RelValTTbar_14TeV             2024
                    12846.0,    # RelValZEE_13                  2024
                    13034.0,    # RelValTTbar_14TeV             2024 PU = Run3_Flat55To75_PoissonOOTPU
                    12834.7,    # RelValTTbar_14TeV             2024 mkFit
                    14034.0,    # RelValTTbar_14TeV             Run3_2023_FastSim 
                    14234.0,    # RelValTTbar_14TeV             Run3_2023_FastSim   PU = Run3_Flat55To75_PoissonOOTPU
                    2500.4,     # RelValTTbar_14TeV             NanoAOD from existing MINI

                    # Phase2
                    24834.0,    # RelValTTbar_14TeV                     phase2_realistic_T25        Extended2026D98         (Phase-2 baseline)   
                    24834.911,  # TTbar_14TeV_TuneCP5                   phase2_realistic_T25        DD4hepExtended2026D98   DD4Hep (HLLHC14TeV BeamSpot) 
                    25034.999,  # RelValTTbar_14TeV (PREMIX)            phase2_realistic_T25        Extended2026D98         AVE_50_BX_25ns_m3p3     
                    24896.0,    # RelValCloseByPGun_CE_E_Front_120um    phase2_realistic_T25        Extended2026D98
                    24900.0,    # RelValCloseByPGun_CE_H_Coarse_Scint   phase2_realistic_T25        Extended2026D98  
                    23234.0,    # TTbar_14TeV_TuneCP5                   phase2_realistic_T21        Extended2026D94         (exercise with HFNose) 
                    

                    ###### pp Data
                    ## Run1
                    4.22,       # Run2011A  Cosmics 
                    4.53,       # Run2012B  Photon                      miniAODs
                    1000,       # Run2011A  MinimumBias Prompt          RecoTLR.customisePrompt
                    1001,       # Run2011A  MinimumBias                 Data+Express
                    ## Run2
                    136.731,    # Run2016B SinglePhoton  
                    136.7611,   # Run2016E JetHT (reMINIAOD)            Run2_2016_HIPM + run2_miniAOD_80XLegacy
                    136.8311,   # Run2017F JetHT (reMINIAOD)            run2_miniAOD_94XFall17
                    136.88811,  # Run2018D JetHT (reMINIAOD)            run2_miniAOD_UL_preSummer20 (UL MINI)
                    136.793,    # Run2017C DoubleEG                      
                    136.874,    # Run2018C EGamma
                     
                    ## Run3
                    # 2021
                    139.001,    # Run2021  MinimumBias                  Commissioning2021   
                    
                    # 2022
                    140.023,    # Run2022B ZeroBias 
                    140.043,    # Run2022C ZeroBias 
                    140.063,    # Run2022D ZeroBias 

                    # 2023
                    141.044,    # Run2023D JetMET0
                    141.042,    # Run2023D ZeroBias
                    141.046,    # Run2023D EGamma0

                    ###### Heavy Ions
                    ## Data
                    # Run2   
                    140.56,    # HIRun2018A HIHardProbes                    Run2_2018_pp_on_AA 
                    ## MC
                    158.01,    # RelValHydjetQ_B12_5020GeV_2018_ppReco      (reMINIAOD) (HI MC with pp-like reco)
                    312.0,     # Pyquen_ZeemumuJets_pt10_2760GeV            PU : HiMixGEN 

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

    parser.add_argument('--nEvents',
                        help='number of events to process in cmsRun. If 0 will use the standard 10 events.',
                        dest='nEvents',
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

    parser.add_argument('-f','--failed-from',
                        help='Provide a matrix report to specify the workflows to be run again. Augments the -l option if specified already',
                        dest='failed_from',
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
    if opt.failed_from:
        rerunthese=[]
        with open(opt.failed_from,'r') as report:
            for report_line in report:
                if 'FAILED' in report_line:
                    to_run,_=report_line.split('_',1)
                    rerunthese.append(to_run)
        if opt.testList:
            opt.testList+=','.join(['']+rerunthese)
        else:
            opt.testList = ','.join(rerunthese)

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
        from colorama import Fore, Style
        from os import isatty
        import subprocess
        import time

        class TheMatrix(cmd.Cmd):
            intro = "Welcome to the Matrix (? for help)"
            prompt = "matrix> "

            def __init__(self, opt):
                cmd.Cmd.__init__(self)
                self.opt_ = opt
                self.matrices_ = {}
                tmp = MatrixReader(self.opt_)
                self.processes_ = dict()
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
                else:
                    is_pipe = not isatty(sys.stdin.fileno())
                    print(Fore.RED + "Error: " + Fore.RESET + "unrecognized command.")
                    # Quit only if given a piped command.
                    if is_pipe:
                      sys.exit(1)

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
                    print("[ " + Fore.RED + ", ".join([str(k) for k in predefinedSet.keys()]) + Fore.RESET + " ]")

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
                        print(Fore.RED + Style.BRIGHT + k)
                    print(Style.RESET_ALL)
                else:
                    selected = arg.split()
                    for k in selected:
                        if k not in self.matrices_.keys():
                            print("Unknown workflow %s: skipping" % k)
                        else:
                            for wfl in self.matrices_[k].workFlows:
                                print("%s %s" % (Fore.BLUE + str(wfl.numId) + Fore.RESET,
                                                              Fore.GREEN + wfl.nameId + Fore.RESET))
                            print("%s contains %d workflows" % (Fore.RED + k + Fore.RESET, len(self.matrices_[k].workFlows)))

            def do_runWorkflow(self, arg):
                # Split the input arguments into a list
                args = arg.split()
                if len(args) < 2:
                    print(Fore.RED + Style.BRIGHT + "Wrong number of parameters passed")
                    print(Style.RESET_ALL)
                    return
                workflow_class = args[0]
                workflow_id = args[1]
                passed_down_args = list()
                if len(args) > 2:
                  passed_down_args = args[2:]
                print(Fore.YELLOW + Style.BRIGHT + "Running with the following options:\n")
                print(Fore.GREEN + Style.BRIGHT + "Workflow class: {}".format(workflow_class))
                print(Fore.GREEN + Style.BRIGHT + "Workflow ID:    {}".format(workflow_id))
                print(Fore.GREEN + Style.BRIGHT + "Additional runTheMatrix options: {}".format(passed_down_args))
                print(Style.RESET_ALL)
                if workflow_class not in self.matrices_.keys():
                    print(Fore.RED + Style.BRIGHT + "Unknown workflow selected: {}".format(workflow_class))
                    print("Available workflows:")
                    for k in self.matrices_.keys():
                         print(Fore.RED + Style.BRIGHT + k)
                    print(Style.RESET_ALL)
                    return
                wflnums = [x.numId for x in self.matrices_[workflow_class].workFlows]
                if float(workflow_id) not in wflnums:
                    print(Fore.RED + Style.BRIGHT + "Unknown workflow {}".format(workflow_id))
                    print(Fore.GREEN + Style.BRIGHT)
                    print(wflnums)
                    print(Style.RESET_ALL)
                    return
                if workflow_id in self.processes_.keys():
                    # Check if the process is still active
                    if self.processes_[workflow_id][0].poll() is None:
                        print(Fore.RED + Style.BRIGHT + "Workflow {} already running!".format(workflow_id))
                        print(Style.RESET_ALL)
                        return
                # If it was there but it's gone, proceeed and update the value for the same key
                # run a job, redirecting standard output and error to files
                lognames = ['stdout', 'stderr']
                logfiles = tuple('%s_%s_%s.log' % (workflow_class, workflow_id, name) for name in lognames)
                stdout = open(logfiles[0], 'w')
                stderr = open(logfiles[1], 'w')
                command = ('runTheMatrix.py', '-w', workflow_class, '-l', workflow_id)
                if len(passed_down_args) > 0:
                  command += tuple(passed_down_args)
                print(command)
                p = subprocess.Popen(command,
                    stdout = stdout,
                    stderr = stderr)
                self.processes_[workflow_id] = (p, time.time())


            def complete_runWorkflow(self, text, line, start_idx, end_idx):
                if text and len(text) > 0:
                    return [t for t in self.matrices_.keys() if t.startswith(text)]
                else:
                    return self.matrices_.keys()

            def help_runWorkflow(self):
              print("\n".join(["runWorkflow workflow_class workflow_id\n",
                "This command will launch a new and independent process that invokes",
                "the command:\n",
                "runTheMatrix.py -w workflow_class -l workflow_id [runTheMatrix.py options]",
                "\nYou can specify just one workflow_class and workflow_id per invocation.",
                "The job will continue even after quitting the interactive session.",
                "stdout and stderr of the new process will be automatically",
                "redirected to 2 logfiles whose names contain the workflow_class",
                "and workflow_id. Mutiple command can be issued one after the other.",
                "The working directory of the new process will be the directory",
                "from which the interactive session has started.",
                "Autocompletion is available for workflow_class, but",
                "not for workflow_id. Supplying a wrong workflow_class or",
                "a non-existing workflow_id for a valid workflow_class",
                "will trigger an error and no process will be invoked.",
                "The interactive shell will keep track of all active processes",
                "and will prevent the accidental resubmission of an already",
                "active jobs."]))

            def do_jobs(self, args):
                print(Fore.GREEN + Style.BRIGHT + "List of jobs:")
                for w in self.processes_.keys():
                    if self.processes_[w][0].poll() is None:
                      print(Fore.YELLOW + Style.BRIGHT + "Active job: {} since {:.2f} seconds.".format(w, time.time() - self.processes_[w][1]))
                    else:
                        print(Fore.RED + Style.BRIGHT + "Done job: {}".format(w))
                print(Style.RESET_ALL)

            def help_jobs(self):
              print("\n".join(["Print a full list of active and done jobs submitted",
                "in the ongoing interactive session"]))

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
                    if re.match(pattern, wfl.nameId):
                      print("%s %s" % (Fore.BLUE + str(wfl.numId) + Fore.RESET,
                                       Fore.GREEN + wfl.nameId + Fore.RESET))
                      counter +=1
                print("Found %s compatible workflows inside %s" % (Fore.RED + str(counter) + Fore.RESET,
                                                                   Fore.YELLOW + str(args[0])) + Fore.RESET)

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

                fmt   = "[%s]: %s\n"
                maxLen = 100
                for wflid in wflids:
                    dump = True
                    for key, mrd in self.matrices_.items():
                        for wfl in mrd.workFlows:
                            if wfl.numId == float(wflid):
                                if dump:
                                    dump = False
                                    print(Fore.GREEN + str(wfl.numId) + Fore.RESET + " " + Fore.YELLOW + wfl.nameId + Fore.RESET)
                                    for i,s in enumerate(wfl.cmds):
                                        print(fmt % (Fore.RED + str(i+1) + Fore.RESET,
                                          (str(s)+' ')))
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
